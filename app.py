from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/flask')
def hello_world():
    return 'Hello World!'


@app.route('/find_similar_users', methods=['POST'])
def recommend():
    """
    param:
    - liked_reviews: 123,7,44,25,6
    return:
    - 게시글을 리턴할건지? 유사한 사용자 리스트를 리턴할건지?
    """
    print(request.data)
    liked_reviews = request.data['liked_reviews']
    print(liked_reviews)
    similar_users = find_similar_users(liked_reviews)  # TODO !!!!
    return jsonify({"similar_users": similar_users})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
