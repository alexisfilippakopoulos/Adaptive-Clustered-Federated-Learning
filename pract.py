from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np
import sqlite3
import pickle
import os

pretrained_clients = [1, 2]
def execute_query(query: str, values=None, fetch_data_flag=False, fetch_all_flag=False):
    try:
        connection = sqlite3.Connection('server_data/server_db.db')
        cursor = connection.cursor()
        cursor.execute(query, values) if values is not None else cursor.execute(query)
        fetched_data = (cursor.fetchall() if fetch_all_flag else cursor.fetchone()[0]) if fetch_data_flag else None
        connection.commit()
        connection.close()        
        return fetched_data
    except sqlite3.Error as error:
        print(f'{query} \nFailed with error:\n{error}')

def calculate_l2_norm_fc(weights1, weights2):
    params1 = [p1.cpu().detach().numpy().flatten() for name, p1 in weights1.items() if name.__contains__('fc')]
    params2 = [p2.cpu().detach().numpy().flatten() for name, p2 in weights2.items() if name.__contains__('fc')]

    diff_norm = np.linalg.norm(np.concatenate([(p1 - p2).flatten() for p1, p2 in zip(params1, params2)]), ord=2)
    return diff_norm


def example():
    query = "SELECT model_updated_weights FROM training WHERE client_id IN (" + ", ".join(str(id) for id in pretrained_clients) + ") AND epoch = -1"
    print(query)
    all_pretrained_weights = execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
    print(all_pretrained_weights.__class__)
    print(len(all_pretrained_weights))
    print(all_pretrained_weights[0].__class__)
    print(all_pretrained_weights[0][0].__class__)
    weights = pickle.loads(all_pretrained_weights[0][0])
    print(weights.keys())
    [(print(params), print(name)) for name, params in weights.items()]
    params = [p1.cpu().detach().numpy().flatten() for name, p1 in weights.items() if name.__contains__('fc')]
    print(len(params))
    '''fc_sims = np.full((len(pretrained_clients), len(pretrained_clients)), np.nan)
    for i in range(len(pretrained_clients)):
        for j in range(len(pretrained_clients)):
            similarity = calculate_l2_norm_fc(models[i], models[j]).item()
            fc_sims[i][j] = similarity'''
    
def example1():
    query = "SELECT model_updated_weights FROM training WHERE client_id = ? AND epoch = ?"
    client_weights = pickle.loads(execute_query(query=query, values=(1, 1), fetch_data_flag=True))
    print(client_weights.__class__)
    print(client_weights.keys())


def create_clusters(distance_threshold):
    query = "SELECT model_updated_weights FROM training WHERE client_id IN (" + ", ".join(str(id) for id in pretrained_clients) + ") AND epoch = -1"
    all_pretrained_weights = execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
    fc_sims = np.full((len(all_pretrained_weights), len(all_pretrained_weights)), np.nan)
    for i in range(len(all_pretrained_weights)):
        for j in range(len(all_pretrained_weights)):
            similarity = calculate_l2_norm_fc(pickle.loads(all_pretrained_weights[i][0]), pickle.loads(all_pretrained_weights[j][0])).item()
            fc_sims[i][j] = similarity
    
    cluster_ids = fcluster(linkage(squareform(fc_sims)), t=distance_threshold, criterion='distance')
    print(fc_sims)
    print(cluster_ids)
    for i in range(len(pretrained_clients)):
        query = "UPDATE clients SET cluster_id = ? WHERE id = ?"
        execute_query(query=query, values=(int(cluster_ids[i]), pretrained_clients[i]))
    
    query = "SELECT id, cluster_id FROM clients"
    results = execute_query(query=query, fetch_data_flag=True, fetch_all_flag=True)
    for row in results:
        print(row)
    

if __name__ == '__main__':
    print(os.getcwd())

    print(os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir), 'data/'))
    #create_clusters(0.7)