import socket
import threading
import pickle
import sys
import torch.nn as nn
import sqlite3
import torch
from fl_strategy import FL_Strategy
from fl_plan import FL_Plan
from client_model import ClientModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
import time


# Events to ensure synchronization
weights_recvd_event = threading.Event()
client_recvd_event = threading.Event()

CURRENT_EPOCH = -1
DISTANCE_THRESHOLD = 0.9
LAMBDA = 0.5

class Server:
    def __init__(self, server_ip, server_port):
        self.connected_clients = {}
        self.trained_clients = []
        self.pretrained_clients = []
        self.cluster_dict = {}
        self.ip = server_ip
        self.port = int(server_port)
        self.server_db_path = 'server_data/server_db.db'
        self.event_dict = {'UPDATED_WEIGHTS': weights_recvd_event, 'OK': client_recvd_event}
        self.device = self.get_device()
        print(f'Using {self.device}')
        torch.manual_seed(32)
        self.client_model = ClientModel()
        self.recvd_initial_weights = 0

    def create_db_schema(self):
        """
        Creates the server-side database schema.
        """
        clients_table = """
        CREATE TABLE clients(
            id INT PRIMARY KEY,
            ip VARCHAR(50),
            port INT,
            datasize INT,
            cluster_id INT
        )
        """
        training_table = """
        CREATE TABLE training(
            client_id INT,
            epoch INT,
            model_updated_weights BLOB,
            model_aggregated_weights BLOB,
            PRIMARY KEY (client_id, epoch),
            FOREIGN KEY (client_id) REFERENCES clients (id)
        )
        """
        epoch_stats_table = """
        CREATE TABLE epoch_stats(
            epoch INT PRIMARY KEY,
            connected_clients INT,
            trained_clients INT
        )
        """
        self.execute_query(query=clients_table) if not self.check_table_existence(target_table='clients') else None
        self.execute_query(query=training_table) if not self.check_table_existence(target_table='training') else None
        self.execute_query(query=epoch_stats_table) if not self.check_table_existence(target_table='epoch_stats') else None
        print('[+] Database schema created/loaded successsfully')

    def check_table_existence(self, target_table: str) -> bool:
        """
        Checks if a specific table exists within the database.
        Args:
            target_table: Table to look for.
        Returns:
            True or False depending on existense.
        """
        query = "SELECT name FROM sqlite_master WHERE type ='table'"
        tables = self.execute_query(query=query, values=None, fetch_data_flag=True, fetch_all_flag=True)
        exists = any(table[0] == target_table for table in tables) if tables is not None else False
        return exists
    
    def execute_query(self, query: str, values=None, fetch_data_flag=False, fetch_all_flag=False):
        """
        Executes a given query. Either for retrieval or update purposes.
        Args:
            query: Query to be executed
            values: Query values
            fetch_data_flag: Flag that signals a retrieval query
            fetch_all_flag: Flag that signals retrieval of all table data or just the first row.
        Returns:
            The data fetched for a specified query. If it is not a retrieval query then None is returned. 
        """
        try:
            connection = sqlite3.Connection(self.server_db_path)
            cursor = connection.cursor()
            cursor.execute(query, values) if values is not None else cursor.execute(query)
            fetched_data = (cursor.fetchall() if fetch_all_flag else cursor.fetchone()[0]) if fetch_data_flag else None
            connection.commit()
            connection.close()        
            return fetched_data
        except sqlite3.Error as error:
            print(f'{query} \nFailed with error:\n{error}')

    def create_socket(self):
        """
        Binds the server-side socket to enable communication.
        """
        try:
            self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.server_socket.bind((self.ip, self.port))
            print(f'[+] Server initialized successfully at {self.ip, self.port}')
        except socket.error as error:
            print(f'Socket initialization failed with error:\n{error}')
            sys.exit(0)

    def listen_for_connections(self):
        """
        Listening to the server-side port for incoming connections from clients.
        Creates a unique communication thread for each connected client. 
        """
        try:
            self.server_socket.listen()
            while True:
                client_socket, client_address = self.server_socket.accept()
                client_id = self.handle_connections(client_address, client_socket)
                threading.Thread(target=self.listen_for_messages, args=(client_socket, client_id)).start()
        except socket.error as error:
            print(f'Connection handling thread failed:\n{error}')
            
    def listen_for_messages(self, client_socket: socket.socket, client_id: int):
        """
        Client-specific communication thread. Listens for incoming messages from a unique client.
        Args:
            client_socket: socket used from a particular client to establish communication.
        """
        data_packet = b''
        try:
            while True:
                data_chunk = client_socket.recv(4096)
                data_packet += data_chunk
                if (b'<END>'in data_packet) and (b'<START>' in data_packet):
                        threading.Thread(target=self.handle_data, args=(data_packet, client_id)).start()
                        data_packet = b'' 
                if not data_chunk:
                    break
        except socket.error as error:
            # Handle client dropout
            print(f'Error receiving data from {client_id, self.connected_clients[client_id][0]}:\n{error}')
            client_socket.close()
            self.connected_clients.pop(client_id)
            self.trained_clients.remove(client_id) if client_id in self.trained_clients else None

    
    def handle_connections(self, client_address: tuple, client_socket: socket.socket):
        """
        When a client connects -> Add him on db if nonexistent, append to connected_clients list and transmit initial weights
        Args: Tuple (client_ip, client_port)
        """
        client_ip, client_port = client_address
        query = """
        SELECT id
        FROM clients
        WHERE ip = ? AND port = ?
        """
        exists = self.execute_query(query=query, values=(client_ip, client_port), fetch_data_flag=True, fetch_all_flag=True)
        if len(exists) == 0:
            query = """
            SELECT id FROM clients ORDER BY id DESC LIMIT 1;
            """
            last_id = self.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
            client_id = 1 if len(last_id) == 0 else last_id[0][0] + 1
            query = """
            INSERT INTO clients (id, ip, port) VALUES (?, ?, ?)
            """
            self.execute_query(query=query, values=(client_id, client_ip, client_port))
        else:
            client_id = exists[0][0]
        self.connected_clients[client_id] = (client_address, client_socket)
        print(f'[+] Client {client_id, client_address} connected -> Connected clients: {len(self.connected_clients)}')
        #self.send_packet(data={'INITIAL_WEIGHTS': [self.client_model.state_dict(), self.classifier_model.state_dict()]}, client_socket=client_socket)
        self.send_packet(data={'PLAN': self.plan}, client_socket=client_socket)
        print(f'[+] Transmitted FL plan to client {client_id, client_address}')
        return client_id

    def send_packet(self, data: dict, client_socket: socket.socket):
        """
        Packs and sends a payload of data to a specified client.
        The format used is <START>DATA<END>, where DATA is a dictionary whose key is the header and whose value is the payload.
        Args:
            data: payload of data to be sent.
            client_socket: socket used for the communication with a specific client.
        """
        try:
            client_socket.sendall(b'<START>' + pickle.dumps(data) + b'<END>')
        except socket.error as error:
            print(f'Message sending failed with error:\n{error}')
            client_socket.close()

    def handle_data(self, data: dict, client_id: int):
        """
        Handles a received data packet according to its contents.
        A packet can be either be:
            1. Model Outputs during labeling process
            2. Updated model weights during training
            3. Signal that initial weights are received
        Args:
            data: Dictionary where the key is the header and the value is the payload
            client_id: The id of the sending client
        """
        # Get payload and header
        data = pickle.loads(data.split(b'<START>')[1].split(b'<END>')[0])
        header = list(data.keys())[0]
        if header == 'UPDATED_WEIGHTS':
            query = """
            INSERT INTO training (client_id, epoch, model_updated_weights) VALUES (?, ?, ?)
            ON CONFLICT (client_id, epoch) DO
            UPDATE SET model_updated_weights = ?"""
            serialized_model_weights = pickle.dumps(data[header])
            self.execute_query(query=query, values=(client_id, CURRENT_EPOCH, serialized_model_weights, serialized_model_weights))
            self.trained_clients.append(client_id)
            print(f"\t[+] Received updated weights of client: {client_id, self.connected_clients[client_id][0]}")
            print(f"\t[+] Currently trained clients: {len(self.trained_clients)} / {self.strategy.MIN_PARTICIPANTS_FIT}")
        elif header == 'PRETRAINED_WEIGHTS':
            query = """
            INSERT INTO training (client_id, epoch, model_updated_weights) VALUES (?, ?, ?)
            ON CONFLICT (client_id, epoch) DO
            UPDATE SET model_updated_weights = ?"""
            serialized_model_weights = pickle.dumps(data[header][0])
            self.execute_query(query=query, values=(client_id, CURRENT_EPOCH, serialized_model_weights, serialized_model_weights))

            query = "UPDATE clients SET datasize = ? WHERE id = ?"
            self.execute_query(query=query, values=(data[header][1], int(client_id)))
            self.pretrained_clients.append(client_id)
            print(f"\t[+] Received pre-trained weights of client: {client_id, self.connected_clients[client_id][0]}")
            print(f"\t[+] Currently pre-trained clients: {len(self.pretrained_clients)} / {self.strategy.MIN_PARTICIPANTS_FIT}")
            
        self.event_dict[header].set() if header in self.event_dict.keys() else None

    def initialize_strategy(self, config_file_path: str):
        """
        Initializes the FL Strategy and FL Plan objects based on the configuration file.
        Args:
            config_file_path: The path to the configuration file
        """
        self.strategy = FL_Strategy(config_file=config_file_path)
        self.plan = FL_Plan(epochs=self.strategy.GLOBAL_TRAINING_ROUNDS, pre_epochs=self.strategy.PRETRAIN_ROUNDS, lr=self.strategy.LEARNING_RATE,
                            loss=self.strategy.CRITERION, optimizer=self.strategy.OPTIMIZER, batch_size=self.strategy.BATCH_SIZE,
                            model_weights=self.client_model.state_dict())
        print(f"[+] Emloyed Strategy:\n{self.strategy}")
        
    def get_device(self):
        """
        Check available devices (cuda or cpu). If cuda then cuda else cpu
        Returns: A torch.device() Object
        """
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def personalized_aggregation(self):
        """
        Aggregation process at the end of each global training round. Fetch necessary data and aggregate the global model by the Federated Averaging algorithm.\
        Returns:
            A list containing [The Federated Averaged client model, The Federated Averaged client classifier]
        """
        for cluster_id, client_ids in self.cluster_dict.items():
            cluster_global_model = self.federated_averaging(needed_clients=client_ids)
            for client_id in client_ids:
                aggr_weights = {}
                # Aggr
                query = "SELECT model_updated_weights FROM training WHERE client_id = ? AND epoch = ?"
                client_weights = pickle.loads(self.execute_query(query=query, values=(client_id, CURRENT_EPOCH), fetch_data_flag=True))
                for key in client_weights.keys():
                        aggr_weights[key] = LAMBDA * client_weights[key] + (1 - LAMBDA) * cluster_global_model[key]     
                # Send
                self.send_packet(data={'AGGR_MODEL': aggr_weights}, client_socket=self.connected_clients[client_id][1])
                # Save on db
                query = "UPDATE training SET model_aggregated_weights = ? WHERE client_id = ? AND epoch = ?"
                self.execute_query(query=query, values=(pickle.dumps(aggr_weights), client_id, CURRENT_EPOCH))
                print([f'[+]Aggregated and transmitted personalized weights for client: {client_id, self.connected_clients[client_id][0]}'])
        return
        

    def federated_averaging(self, needed_clients: list):
        """
        Implementation of the federated averaging algotithm.
        Args:
            fetched_weights: The model weights of the participating/trained clients
            fetched_datasizes: The data sizes of the participating/trained clients
        Returns:
            avg_weights: The global model by Federated Averaging
        """
        # Fetch the updated model weights of all trained clients
        query = "SELECT model_updated_weights FROM training WHERE client_id IN (" + ", ".join(str(id) for id in needed_clients) + ") AND epoch = ?"
        all_client_model_weights = self.execute_query(query=query, values=(CURRENT_EPOCH, ), fetch_data_flag=True, fetch_all_flag=True)
        # Fetch the datasizes of all trained clients
        query = "SELECT datasize FROM clients WHERE id IN (" + ", ".join(str(id) for id in needed_clients) + ")"
        datasizes = self.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
        datasizes = [int(row[0]) for row in datasizes]
        # Dictionary for the global (averaged) weights
        avg_weights = {}
        # Calculate the data size of all the participating clients
        total_data = sum(datasize for datasize in datasizes)
        # For each client's updated weights
        for i in range(len(all_client_model_weights)):
            client_weight_dict = pickle.loads(all_client_model_weights[i][0])
            # For each layer of the model
            for key in client_weight_dict.keys():
                # Average the weights and normalize based on client's contribution to total data size
                if key in avg_weights.keys():
                    avg_weights[key] += client_weight_dict[key] * (datasizes[i] / total_data)
                else:
                    avg_weights[key] = client_weight_dict[key] * (datasizes[i] / total_data)
        return avg_weights
    
    def calculate_l2_norm_fc(self, weights1, weights2):
        # Get the weights of the FC layers only
        params1 = [p1.cpu().detach().numpy().flatten() for name, p1 in weights1.items() if name.__contains__('fc')]
        params2 = [p2.cpu().detach().numpy().flatten() for name, p2 in weights2.items() if name.__contains__('fc')]
        # Calculate model similarity based on the l2 norm of their weights' difference
        diff_norm = np.linalg.norm(np.concatenate([(p1 - p2).flatten() for p1, p2 in zip(params1, params2)]), ord=2)
        return diff_norm

    def create_clusters(self):
        """
        Agglomerative clustering
        """
        # Get the clients' pre-trained weights
        query = "SELECT model_updated_weights FROM training WHERE client_id IN (" + ", ".join(str(id) for id in self.pretrained_clients) + ") AND epoch = -1"
        all_pretrained_weights = self.execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
        # Calculate model similarity matrix based on l2 norm between all client pairs
        fc_sims = np.full((len(all_pretrained_weights), len(all_pretrained_weights)), np.nan)
        for i in range(len(all_pretrained_weights)):
            for j in range(len(all_pretrained_weights)):
                similarity = self.calculate_l2_norm_fc(pickle.loads(all_pretrained_weights[i][0]), pickle.loads(all_pretrained_weights[j][0])).item()
                fc_sims[i][j] = similarity
        # Agglomerative Clustering to obtain cluster ids
        cluster_ids = fcluster(linkage(squareform(fc_sims)), t=DISTANCE_THRESHOLD, criterion='distance')
        # Update client cluster ids on db
        for i in range(len(self.pretrained_clients)):
            query = "UPDATE clients SET cluster_id = ? WHERE id = ?"
            self.execute_query(query=query, values=(int(cluster_ids[i]), self.pretrained_clients[i]))
        # For each cluster find the corresponding clients
        for cluster_id in set(cluster_ids):
            needed_clients = []
            for i in range(len(self.pretrained_clients)):
                if cluster_ids[i] == cluster_id:
                    needed_clients.append(self.pretrained_clients[i])
            self.cluster_dict[cluster_id] = needed_clients
            # Perform FedAvg to obtain initial cluster-level weights
            initial_cluster_weights = self.federated_averaging(needed_clients=needed_clients)
            # Transmit initial cluster-level weights to corresponding clients
            for client_id in needed_clients:
                self.send_packet(data={'AGGR_MODEL': initial_cluster_weights}, client_socket=self.connected_clients[client_id][1])
            print(f'[+] Transmitted initial cluster weights to cluster {cluster_id} for client_ids: {needed_clients}')
        print(self.cluster_dict)
        return

    def get_test_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ), )])
        test_data = datasets.FashionMNIST(root='Implementation/data/', download=True, train=False, transform=transform)
        return DataLoader(dataset=test_data, batch_size=self.strategy.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    def test_global_model(self, test_dl: DataLoader, model_weights: dict, classifier_weights: dict, e: int):
        self.client_model.load_state_dict(model_weights), self.classifier_model.load_state_dict(classifier_weights)
        self.client_model.to(self.device), self.classifier_model.to(self.device)
        self.client_model.eval(), self.classifier_model.eval()
        corr, total = 0, 0
        with torch.inference_mode():
            for i, (inputs, labels) in enumerate(test_dl):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.classifier_model(self.client_model(inputs))
                _, preds = torch.max(input=outputs, dim=1)
                corr += (preds == labels).sum().item()
                total += labels.size(0)
            del inputs, labels
        test_acc = (corr * 100 ) / total
        self.execute_query(query='UPDATE epoch_stats SET test_accuracy = ? WHERE epoch = ?', values=(round(test_acc, 2), e - 1))
        print(f'\t[+] Global model accuracy for epoch {e - 1}: {test_acc} %')

if __name__ == '__main__':
# To execute, server_ip and server_port must be specified from the cl.
    if len(sys.argv) != 3:
        print('Incorrect number of command-line arguments.\nTo execute, server_ip and server_port must be specified from the cl.')
        sys.exit(1)

    server = Server(sys.argv[1], sys.argv[2])
    server.create_socket()
    server.create_db_schema()
    threading.Thread(target=server.listen_for_connections, args=()).start()
    server.initialize_strategy(config_file_path='strategy_config.txt')
    #test_dl = server.get_test_data()

    while (len(server.connected_clients) < server.strategy.MIN_PARTICIPANTS_START) or (len(server.connected_clients) != len(server.pretrained_clients)):
        pass
    
    server.create_clusters()

    for e in range(server.strategy.GLOBAL_TRAINING_ROUNDS):
        CURRENT_EPOCH = e
        # Wait for the minimum number of client to connect and label with the server
        while (len(server.connected_clients) < server.strategy.MIN_PARTICIPANTS_START):
            pass

        print(f'[+] Global training round {e} initiated')
        query = "INSERT INTO epoch_stats (epoch, connected_clients) VALUES (?, ?) ON CONFLICT (epoch) DO UPDATE SET connected_clients = ?"
        server.execute_query(query=query, values=(CURRENT_EPOCH, len(server.connected_clients), len(server.connected_clients)))
        
        # Wait to receive model updates from the minimum number of clients to aggregate
        while len(server.trained_clients) < server.strategy.MIN_PARTICIPANTS_FIT:
            pass
        #Aggregate personalized models
        server.personalized_aggregation()
        server.trained_clients.clear()