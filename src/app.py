"""
Flask webapp main file, start the Flask app server

@author: nidragedd
"""
import argparse
from src import app

if __name__ == '__main__':
    # Handle mandatory arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    ap.add_argument("-p", "--port", type=int, required=True, help="server port number")
    args = vars(ap.parse_args())

    # Launch the Flask app server
    app.run(host=args["ip"], port=args["port"], debug=False, threaded=True, use_reloader=False)
