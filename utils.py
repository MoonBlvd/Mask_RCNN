import csv

def write_csv(file_path, data):
    with open(file_path, 'w') as csvfile:
        # field = [field_name]
        writer = csv.writer(csvfile)
        for i in range(data.shape[0]):
            writer.writerow(data[i])