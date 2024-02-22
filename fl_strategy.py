from collections import OrderedDict


class FL_Strategy:
    """
    The Federated Learning strategy to be used during this training procedure.
    The strategy is configurable via the strategy_config.txt file where we can change the various arguments.
    The strategy object is also used by the server to create the FL_Plan object that is trasmitted to the clients.
    """
    def __init__(self, config_file):
        args = self.read_conf_file(config_file)
        self.MIN_PARTICIPANTS_START = args[0]
        self.MIN_PARTICIPANTS_FIT = args[1]
        self.PRETRAIN_ROUNDS = args[2]
        self.GLOBAL_TRAINING_ROUNDS = args[3]
        self.BATCH_SIZE = args[4]
        self.LEARNING_RATE = args[5]
        self.CRITERION = args[6]
        self.OPTIMIZER = args[7]

    def read_conf_file(self, config_file: str) -> list:
        """
        Reads each line of the 'server_config.txt' file and retrieves the needed information.
        Args:
            config_file: filepath to the config file
        Returns
            attrs: list containing the retrieved info, used to initialize the class's attributes
        """
        attrs = []
        try:
            with open(config_file, 'r') as file:
                for i, line in enumerate(file):
                    if line.__contains__('#'):
                        continue
                    elif not line.__contains__('='):
                        raise Exception(f"Invalid configuration line (missing '=') at\nLine {i + 1}: {line}")
                    else:
                        args = line.strip().lower().split('=')
                        if '' in args:
                            raise Exception(f"Invalid configuration line (missing key or value) at\nLine {i + 1}: {line}")
                        arg = self.decode_args(args)
                        attrs.append(arg)
            return attrs
        except Exception as e:
            print(e)

    def decode_args(self, args: list):
        """
        Process each line of the document and extract the information needed
        Args:
            args = list of parameter name and parameter value
        Returns:
            The value formatted to the correct type.
        """
        num_args = ['min_clients_start', 'min_clients_fit', 'global_epochs', 'batch_size', 'pretrain_epochs']
        param, val = args[0].strip(), args[1].strip()
        if param == 'learning_rate':
            return float(val)
        elif param in num_args:
            return int(val)
        else:
            return val

    def __repr__(self) -> str:
        header = "FL Strategy"
        content = (
            f"MINIMUM CONNECTED CLIENTS: {self.MIN_PARTICIPANTS_START}\n"
            f"MINIMUM CLIENTS TO AGGREGATE: {self.MIN_PARTICIPANTS_FIT}\n"
            f"PRE-TRAINING ROUNDS: {self.PRETRAIN_ROUNDS}\n"
            f"GLOBAL TRAINING ROUNDS: {self.GLOBAL_TRAINING_ROUNDS}\n"
            f"BATCH SIZE: {self.BATCH_SIZE}\n"
            f"LEARNING RATE: {self.LEARNING_RATE}\n"
            f"CRITERION: {self.CRITERION.upper()}\n"
            f"OPTIMIZER: {self.OPTIMIZER.upper()}"
        )
        line_length = max(len(header), len(content.split('\n')[0]))
        
        formatted_output = f"{header.center(line_length, '-')} \n{content}\n{'-'.center(line_length, '-')}"
        return formatted_output
    