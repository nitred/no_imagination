"""Entry point for the server application."""

import json
import logging
import traceback
from no_imagination.generate import GenerateMnist, GenerateFlowers

from flask import Response, redirect, request, send_from_directory
from flask_security import auth_token_required, utils
# from gevent.wsgi import WSGIServer
from pprint import pprint, pformat

from .app_utils import html_codes, token_login
from .factory import create_app, create_user
import gc


is_mnist = False
is_flowers = False
generate_mnist = None
generate_flowers = None
logger = logging.getLogger(__name__)

def activate_model(model_name):
    global is_mnist
    global is_flowers
    global generate_mnist
    global generate_flowers
    logger.info("Activating Model: {}".format(model_name))
    if model_name == "mnist":
        if is_flowers:
            is_flowers = False
            generate_flowers.close()
            del generate_flowers
            gc.collect()
            generate_flowers = None

        if not is_mnist:
            is_mnist = True
            generate_mnist = GenerateMnist()

    else:
        if is_mnist:
            is_mnist = False
            generate_mnist.close()
            del generate_mnist
            gc.collect()
            generate_mnist = None

        if not is_flowers:
            is_flowers = True
            generate_flowers = GenerateFlowers()

app = create_app()


@app.before_first_request
def init():
    """Initialize the application with defaults."""
    logger.info("Creating database")
    create_user(app)


@app.route('/')
def root():
    logger.info("route: /")
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def send_static(path):
    logger.info("route: {}".format(path))
    return send_from_directory('static', path)


@app.route('/api/post/<path:path>', methods=['POST', 'GET'])
def post_form(path):
    result = request.form
    result = json.dumps(result)
    result = json.loads(result)
    logger.info("======================================")
    logger.info(pformat(result))
    logger.info("--------------------------------------")
    text_description = result['text_description']

    if path == "mnist":
        logger.info("mnist")
        logger.info("TEXT Description: {}".format(text_description))
        activate_model("mnist")
        image_name = generate_mnist.generate(text_description)

    elif path == "flowers":
        logger.info("flowers")
        logger.info("TEXT Description: {}".format(text_description))
        activate_model("flowers")
        if text_description is None or text_description == "":
            text_description = "red flower"
        image_name = generate_flowers.generate(text_description)

    else:
        image_name = "not_found"

    return redirect("http://207.154.233.16:4444/img/{}/{}".format(path, image_name), code=302)


@app.route('/generate/<path:path>')
def generate_images(path):
    logger.info("route: {}".format(path))

    if path == "mnist":
        logger.info("MNIST")
        image_name = generate_mnist.generate()

    elif path == "imagenet":
        logger.info("Imagenet")
        image_name = "something"

    else:
        image_name = "not_found"

    return redirect("http://207.154.233.16:4444/img/{}/{}/".format(path, image_name), code=302)


@app.route('/img/<path:path>')
def send_images(path):
    logger.info("route: {}".format(path))
    return send_from_directory('/home/nitred/.no_imagination/server/', path)


# @app.route("/api/logoutuser", methods=['POST'])
# @auth_token_required
# def logout():
#     """Logout the currently logged in user."""
#     logger.info('Logged out user !!')
#     utils.logout_user()
#     return 'logged out successfully', 200
#
#
# @app.route('/api/loginuser', methods=['POST'])
# def login():
#     """View function for login view."""
#     logger.info('Logged in user')
#     return token_login.login_with_token(request, app)
#
#
# @app.route('/api/getdata', methods=['POST'])
# @auth_token_required
# def get_data():
#     """Get dummy data returned from the server."""
#     data = {'Heroes': ['Hero1', 'Hero2', 'Hero3']}
#     json_response = json.dumps(data)
#     return Response(json_response,
#                     status=html_codes.HTTP_OK_BASIC,
#                     mimetype='application/json')


def main():
    """Main entry point of the app."""
    try:
        # http_server = WSGIServer(('0.0.0.0', 4444),
        #                          app,
        #                          log=logging,
        #                          error_log=logging)
        #
        # http_server.serve_forever()
        app.run(host='0.0.0.0', port=4444)
    except Exception as exc:
        logger.error(exc.message)
        logger.exception(traceback.format_exc())
    finally:
        # Do something here
        pass
