from simple_db.database import SimpleDB

def open_db():
    cui_db = SimpleDB('simple_db/data/cui.json')
    semantic_db = SimpleDB('simple_db/data/semantic.json')
    nonentity_db = SimpleDB('simple_db/data/nonentity.json') # entity that UMLS lookup returns no results
    return cui_db, semantic_db, nonentity_db

def close_db(database_list):
    for db in database_list:
        db.save_data()
