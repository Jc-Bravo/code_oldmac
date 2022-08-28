# from urllib import response
from flask import Flask, request, jsonify
from flask_cors import CORS


STATUS = ["1"]

app = Flask(__name__)

app.config.from_object(__name__)

CORS(app, resources={r"/*":{'origins':"*"}})
# CORS(app, resources={r'/*':{'origins': 'http://localhost:8080',"allow_headers": "Access-Control-Allow-Origin"}})




# hello world route
@app.route('/', methods=['GET'])
def greetings():
    return(STATUS[0])
    
# @app.route('/THUbegin', methods=['GET','POST'])
# def THUbegin():
#     status = "0.0"
#     return (status)
    # request_object = {}
    # if request_method =="POST":
    #     status = 6.6

@app.route('/THUconnect', methods=['GET','POST'])
def THUconnect():
    response_object = {'status':'success'}
    if request.method =="POST":
        # STATUS = "data is 100"
        # STATUS.append({
        #     # 'id' : uuid.uuid4().hex,
        #     # 'title': post_data.get('title'),
        #     # 'genre': post_data.get('genre'),
        #     # 'played': post_data.get('played')
        #     2
        #     })
        response_object['message'] =  'Game Added!'
        STATUS.append("100")
        return STATUS[len(STATUS)-1]
    else:
        # response_object['games'] = STATUS[0]
        return STATUS[len(STATUS)-1]
    # return jsonify(response_object)
    



if __name__ == "__main__":
    app.run(debug=True)