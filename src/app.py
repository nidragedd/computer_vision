"""
Flask webapp main file, start the Flask app server

@author: nidragedd
"""
from src import app
from src.config import config
from src.config.pgconf import ProgramConfiguration

if __name__ == '__main__':
    # Handle mandatory arguments
    args = config.parse_applaunch_args()
    config_file = vars(args)['config_file']
    log_config_file = vars(args)['log_file']

    # Configure the whole program (logging, external config files, singletons, ...)
    config.configure_logging(log_config_file)
    config.pgconf = ProgramConfiguration(config_file)

    # Launch the Flask app server
    app.run(host=vars(args)["ip"], port=vars(args)["port"], debug=False, threaded=True, use_reloader=False)
