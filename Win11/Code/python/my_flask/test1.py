from flask import Flask
from flask import url_for
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return 'index'


@app.route('/user/<username>')
def profile(username):
    return f'{username}\'s profile'


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return "do_the_login()"
    else:
        return "show_the_login_form()"


with app.test_request_context():
    print(url_for('index'))
    print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))

if __name__ == "__main__":
    app.run(debug=True)