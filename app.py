from flask import Flask, request, jsonify
import recommend
import json


app = Flask(__name__)


# @app.route('/flask')
# def hello_world():
#     return 'Hello World!'

@app.route('/find_similar_users', methods=['POST'])
def passId():
    id = request.form['id']
    # print(type(id))
    userList = []
    # recommend.neighbors(recommend.foo().shape[0])
    userList = userList + (recommend.getIdIndex(id))
    print(userList)

    if not userList:
        listToString = 'null'
        print("list To String", listToString)
        return listToString
    else:
        listToString = ', '.join(userList)
        print(userList, type(userList))
        print("find_similar_users : ", userList)
        print("list To String", listToString)

        return listToString


# @app.route('/find_similar_users', methods=['POST'])
# def recommend():
#     """
#     param:
#     - liked_reviews: 123,7,44,25,6
#     return:
#     - 게시글을 리턴할건지? 유사한 사용자 리스트를 리턴할건지?
#     """
#     print(request.data)
#     liked_reviews = request.data['liked_reviews']
#     print(liked_reviews)
#     similar_users = find_similar_users(liked_reviews)  # TODO !!!!
#     return jsonify({"similar_users": similar_users})


if __name__ == '__main__':
    app.run()
