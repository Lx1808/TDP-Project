import mysql.connector
import yaml
import os

def load_database_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../database.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_table():
    config = load_database_config()
    connection = mysql.connector.connect(**config['development'])  # Change to appropriate environment
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            question_text VARCHAR(255),
            question_count INT,
            INDEX idx_question_count (question_count)
        )
    """)
    # 检查 comments 表是否存在，如果不存在则创建
    cursor.execute("SHOW TABLES LIKE 'comments';")
    result = cursor.fetchone()
    if result:
        print("comments Table exists.")
    else:
        print("comments Table does not exist, creating table...")
        cursor.execute("""
            CREATE TABLE comments (
                id INT AUTO_INCREMENT PRIMARY KEY,
                content TEXT
            );
        """)
        print("comments Table created.")

    cursor.close()
    connection.close()

if __name__ == "__main__":
    print("start creating questions table..........")
    create_table()
    print("creating questions table successfully..........")