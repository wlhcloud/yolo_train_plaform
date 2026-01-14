#!/usr/bin/env python
import os
from app import create_app
from managers.rtsp_threadpool_manager import rtsp_manager

if __name__ == "__main__":
    app = create_app()

    with app.app_context():
        rtsp_manager.set_app(app=app)
    app.run(debug=True, host="0.0.0.0", port=5100)
