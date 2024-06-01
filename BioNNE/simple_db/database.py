'''
The simple_db database is a simple database that stores data in a dictionary file. It supports basic CRUD operations.
The database is stored in a dictionary file. The dictionary file is a JSON file that stores the data in the following format:
{
    {
        "key1": "value1",
        "key2": "value2",
        ...
    }
}
'''
import json
import os

class SimpleDB:
    def __init__(self, db_file):
        self.db_file = db_file
        self.data = {}
        self.load_data()

    def load_data(self):
        '''Load data from the database file, return False if database doesn't exist'''
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                self.data = json.load(f)
                return True
        else:
            # Create a database file if it doesn't exist
            self.save_data()
        return False

    def save_data(self):
        with open(self.db_file, 'w') as f:
            json.dump(self.data, f)
    
    def get(self, key):
        return self.data.get(key, None)
    
    def set(self, key, value, save=False):
        self.data[key] = value
        if save:
            self.save_data()
    
    def delete(self, key, save=False):
        if key in self.data:
            del self.data[key]
            if save:
                self.save_data()

    