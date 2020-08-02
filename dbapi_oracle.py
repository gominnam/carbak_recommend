import cx_Oracle

def create_connection():
    # DataSoucreName을 생성
    dsn = cx_Oracle.makedsn("localhost", 49161, "XE")
    db = cx_Oracle.connect("hr2", "1", dsn)
    return db

def basic_query():
    with create_connection() as conn:
        cursor = conn.cursor() # 커서 정보
        #  SQL 실행
        sql = "SELECT reviewlike.id id, reviewlike.reviewno reviewNo, nvl2(reviewlike.id, 1, 0) reviewLike " \
              "FROM readCount right outer join reviewlike " \
              "ON reviewlike.reviewno = readcount.reviewno and reviewlike.id = readcount.id " \
              "UNION " \
              "SELECT readcount.id id, readcount.reviewno reviewno, nvl2(reviewlike.id, 1, 0) reviewLike " \
              "FROM readcount left outer join reviewlike " \
              "ON reviewlike.reviewno = readcount.reviewno and reviewlike.id = readcount.id"
        cursor.execute(sql) # 쿼리 실행
        result = cursor.fetchall()

        return result
