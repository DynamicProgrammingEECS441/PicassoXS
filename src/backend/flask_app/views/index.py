"""
Insta485 index (main) view.

URLs include:
/
"""
import flask
import flask_app


@flask_app.app.route('/upload_img/', methods=['POST'])
def upload_img():
    content_img = flask.request.files['file']
    print(content_img)
    context = {
        "status": "success",
    }
    return flask.jsonify(**context)

    # return flask.render_template("index.html", **context)