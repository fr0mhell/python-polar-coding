from pymongo import MongoClient

username = 'root'
password = 'example'
uri = f'mongodb://{username}:{password}@localhost'
client = MongoClient(uri)
DB_NAME = 'polar_codes_modelling'
