import os
import sys
import time

from dotenv import dotenv_values, find_dotenv, load_dotenv
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
from neo4j import GraphDatabase

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CUR_DIR, ".."))

from log.logutli import Logger

class Neo4jManager:
    def __init__(self, uri, username, password, logger):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._logger = logger
    
    def close(self):
        self.driver.close()

    def import_characters(self, csv_path):
        try:
            with open(csv_path, 'r', encoding='gbk') as file:
                lines = file.readlines()[1:]
                with self.driver.session() as session:
                    for line in lines:
                        name, gender, age, alias, story = line.strip().split(',')  # 注意story不要有换行符
                        # breakpoint()
                        session.run("CREATE (:Character {name: $name, gender: $gender, age: $age, aliases: $alias, background_story: $story})",
                                    name=name, gender=gender, age=age, alias=alias, story=story.replace("\n", " "))

        except FileNotFoundError:
            self._logger.error(f"文件 {csv_path} 不存在！请检查。")
            sys.exit(1)

    def import_relationships(self, csv_path):
        try:
            with open(csv_path, 'r', encoding='gbk') as file:
                lines = file.readlines()[1:]
                with self.driver.session() as session:
                    for line in lines:
                        from_id, to_id, relation = line.strip().split(',')
                        cypher_query = f"MATCH (from:Character {{name: '{from_id}'}}) " \
                                    f"MATCH (to:Character {{name: '{to_id}'}}) " \
                                    f"CREATE (from)-[:{relation}]->(to)"
                        session.run(cypher_query)

        except FileNotFoundError:
            self._logger.error(f"文件 {csv_path} 不存在！请检查。")
            sys.exit(1)

    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def query_status(self):
        with self.driver.session() as session:
            result     = session.run("MATCH (n) RETURN COUNT(n) AS node_count")
            node_count = result.single()["node_count"]

            if node_count == 0:
                self._logger.info("Neo4j 图数据库为空！请检查。")
            else:
                self._logger.info(f"Neo4j 图数据库节点数量：{node_count}，状态正常。")

            return 1 if node_count != 0 else 0
        
    def query_alias(self, person_name):
        with self.driver.session() as session:
            result = session.run("MATCH (person:Character {name: $person_name})"
                                    "RETURN person.alias AS alias",
                                    person_name=person_name)
            alias = result.single()["alias"]
            return alias
            
    def query_relationship(self, person_name, relationship_type):
        with self.driver.session() as session:
            query = f"MATCH (person:Character {{name: '{person_name}'}})-[:{relationship_type}]->(related) \
                        RETURN related.name AS related_name"
            result        = session.run(query)
            related_names = [record["related_name"] for record in result]
            return related_names
        
    def query_person_info(self, person_name):
        with self.driver.session() as session:
            result = session.run("MATCH (person:Character {name: $person_name})"
                                    "RETURN person.name AS name, person.gender AS gender, person.alias AS alias",
                                    person_name=person_name)
            record = result.single()
            if record:
                return {
                    "name": record["name"],
                    "gender": record["gender"],
                    "alias": record["alias"]
                }
            else:
                return None

if __name__ == "__main__":
    # Neo4j 参数
    uri                   = "neo4j+s://5bb87071.databases.neo4j.io"
    username              = "neo4j"
    password              = "PPcEmlN5ognHV-YG0J9Puj0FzCwdiXsElSbaHugN9xA"
    person_csv_path       = './KG/Persons.csv'
    relationship_csv_path = './KG/Relationship.csv'
    env_path              = './KG/.env'  # 环境配置文件，存放OPENAI_API_KEY

    # 初始化日志
    log_id     = 'KG'  
    log_dir    = f'./log/result/'
    log_name   = f'test_KG_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.log'
    log_level  = 'debug'
    logger = Logger(log_id, log_dir, log_name, log_level).logger

    # 初始化代理
    os.environ['HTTPS_PROXY']    = 'http://127.0.0.1:7890'
    os.environ["HTTP_PROXY"]     = 'http://127.0.0.1:7890'

    # 读取本地的环境变量 
    env_vars = dotenv_values(env_path)
    openai_api_key = env_vars['OPENAI_API_KEY']
    # logger.info(f'openai_api_key: {openai_api_key}')

    # 云端创建Neo4j图
    logger.info('Creating Neo4j Graph...')
    manager = Neo4jManager(uri, username, password, logger)
    # manager.clear_database()
    # manager.import_characters(person_csv_path)
    # manager.import_relationships(relationship_csv_path)
    status = manager.query_status()
    # logger.info(f'图数据库状态：（1正常，0不正常）：{status}')

    graph = Neo4jGraph(
        url=uri, username=username, password=password
    )
    logger.info(graph.get_schema)
    logger.info('知识图谱构建完成！')

    # 初始化链路
    chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(temperature=0), graph=graph, verbose=True
    )

    logger.info(chain.run("甄嬛的女儿是谁？"))
    logger.info(chain.run("甄嬛的爸爸是谁？"))
    logger.info(chain.run("华妃的哥哥是谁？"))
    logger.info(chain.run("甄嬛20岁时的称号是？"))
    logger.info(chain.run("甄嬛在30岁时的别名是？"))
    logger.info(chain.run("谁害死了温实初？"))
    logger.info(chain.run("谁杀死了淳常在？"))
    # breakpoint()