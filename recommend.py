import pandas as pd
import numpy as np
import dbapi_oracle as db
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors


result = db.basic_query()
df = pd.DataFrame(result)
df.columns = ["user_id", "reviewNo", "like"]

def foo():
    global user_like

    n_users = df.user_id.unique().shape[0]
    n_reviews = df.reviewNo.unique().shape[0]
    likes = np.zeros((n_users, n_reviews))

    user_like = df.pivot_table('like', index='user_id', columns='reviewNo').fillna(0)

    return user_like.values


def machine():
    global n_like_train, n_like_test, user_pred

    n_like_train, n_like_test = train_test_split(foo(), test_size=0.33, random_state=42)

    cosine_distances(n_like_train)
    distances = 1 - cosine_distances(n_like_train)

    user_pred = distances.dot(n_like_train) / np.array([np.abs(distances).sum(axis=1)]).T


def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def neighbors(input): # 해당 id 탑 유저 목록
    global user_pred_k, top_k_users
    k=input

    result=foo()

    neigh = NearestNeighbors(n_neighbors=k, metric="cosine")
    neigh.fit(result)

    top_k_distances, top_k_users = neigh.kneighbors(result, return_distance=True)

    user_pred_k = np.zeros(result.shape)

    for i in range(result.shape[0]):
        user_pred_k[i, :] = top_k_distances[i].T.dot(result[top_k_users][i]) / \
                            np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T


    return True


def getIdIndex(id): # 로그인한 id의 db 인덱스 값 얻기

    neighbors(foo().shape[0])
    list = []
    for i in range(len(user_like)):
        if id == user_like.index[i]:

            list += topRecommendId(i)

    return list


def topRecommendId(index): # 최종 리턴 함수
    i=0
    list = []

    for i in range(0, 5):
        i = i+1
        list.append(user_like.index[top_k_users[index][i]])

    # print(list)

    return list


# userList = getIdIndex("id22")

if __name__=="__main__":
    # machine()
    # print(foo().shape[0])
    getIdIndex("id22")

    # print(getIdIndex("id30"))
    # print(range(len(user_like)-1))
    # print(top_k_users)
    # neighbors()
    # print(n_like_train.shape)
    # print(n_like_test)
    # print(user_pred_k)
    # print(np.sqrt(get_mse(user_pred_k, n_like_train)))
    # print(np.sqrt(get_mse(user_pred_k, n_like_test)))

