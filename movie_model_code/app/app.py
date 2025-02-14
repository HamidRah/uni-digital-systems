# app.py
from flask import Flask
from app.extensions import db, login_manager
from app.routes import bp

import torch
from .model_architecture_local import MovieRecommender


def create_app(config_name=None):
    app = Flask(__name__)
    if config_name == 'testing':
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False
    else:
        app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://hamid:U9kknW3Q^8@db/moviemanager_db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    app.config['SECRET_KEY'] = 'secret-key'

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'main.login' # type: ignore

    app.register_blueprint(bp)

    # Load the model
    model_path = 'app/ml_model/weighted-optimised.ckpt'
    try:
        model = MovieRecommender.load_from_checkpoint(model_path, map_location='cpu')
        model.eval()
        app.config['model'] = model
    except Exception as e:
        print(f"Failed to load model: {e}")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)

