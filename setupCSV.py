"""
Script che prende il dataset in formato sql e lo trasforma in file csv da 40K righe l'uno
"""
import os
import csv
from tqdm import tqdm


def sql_to_csv_chunks(sql_file_path, csv_output_dir, year, chunk_size=40000):
    # Controlla se il file SQL esiste
    if not os.path.isfile(sql_file_path):
        print(f"Il file SQL {sql_file_path} non esiste.")
        return

    # Crea la directory di output se non esiste
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)

    # Ottieni la dimensione totale del file in byte per la barra di progresso
    total_size = os.path.getsize(sql_file_path)

    # Variabile per accumulare le linee degli statement INSERT
    insert_lines = ''
    in_insert = False
    bytes_read = 0  # Contatore per i byte letti
    record_count = 0  # Contatore per i record
    file_count = 1  # Contatore per i file CSV

    # Apri il file SQL in modalità lettura
    with open(sql_file_path, 'r', encoding='utf-8') as sql_file:
        # Inizializza la barra di progresso
        with tqdm(total=total_size, desc='Conversione in corso', unit='byte') as pbar:
            for line in sql_file:
                stripped_line = line.strip()
                # Aggiorna la barra di progresso
                bytes_read += len(line.encode('utf-8'))
                pbar.update(len(line.encode('utf-8')))

                # Inizia a raccogliere le linee se trova un INSERT INTO
                if stripped_line.startswith('INSERT INTO'):
                    in_insert = True
                    insert_lines = stripped_line
                    # Se l'INSERT termina sulla stessa linea
                    if stripped_line.endswith(';'):
                        # Processa lo statement e gestisci i chunk
                        records = parse_insert_statement(insert_lines)
                        for record in records:
                            if record_count % chunk_size == 0:
                                if record_count > 0:
                                    csv_file.close()
                                csv_file_path = os.path.join(
                                    csv_output_dir, f'ais_data_{year}_p_{file_count}.csv')
                                csv_file = open(csv_file_path, 'w', newline='', encoding='utf-8')
                                csv_writer = csv.writer(csv_file)
                                # Scrivi l'intestazione
                                csv_writer.writerow(['id', 'timestamp', 'MMSI', 'Latitude', 'Longitude', 'Distance', 'Bearing', 'Type'])
                                file_count += 1
                            csv_writer.writerow(record)
                            record_count += 1
                        in_insert = False
                elif in_insert:
                    insert_lines += ' ' + stripped_line
                    # Verifica se lo statement INSERT termina
                    if stripped_line.endswith(';'):
                        # Processa lo statement e gestisci i chunk
                        records = parse_insert_statement(insert_lines)
                        for record in records:
                            if record_count % chunk_size == 0:
                                if record_count > 0:
                                    csv_file.close()
                                csv_file_path = os.path.join(
                                    csv_output_dir, f'ais_data_part_{file_count}.csv')
                                csv_file = open(csv_file_path, 'w', newline='', encoding='utf-8')
                                csv_writer = csv.writer(csv_file)
                                # Scrivi l'intestazione
                                csv_writer.writerow(['id', 'timestamp', 'MMSI', 'Latitude', 'Longitude', 'Distance', 'Bearing', 'Type'])
                                file_count += 1
                            csv_writer.writerow(record)
                            record_count += 1
                        in_insert = False

    # Chiudi l'ultimo file CSV aperto
    if 'csv_file' in locals() and not csv_file.closed:
        csv_file.close()

    print(f"Conversione completata. Sono stati creati {file_count - 1} file CSV in {csv_output_dir}")


def parse_insert_statement(insert_statement):
    # Trova l'inizio dei valori
    values_start = insert_statement.find('VALUES') + len('VALUES')
    values_str = insert_statement[values_start:].strip().rstrip(';')

    # Dividi i record
    records = []
    current_record = ''
    in_string = False
    escape_next = False

    for char in values_str:
        if escape_next:
            current_record += char
            escape_next = False
        elif char == '\\':
            escape_next = True
            current_record += char
        elif char == '\'':
            in_string = not in_string
            current_record += char
        elif char == ',' and not in_string:
            current_record += '\t'  # Usa tab come separatore temporaneo
        elif char == '(' and not in_string:
            current_record = ''
        elif char == ')' and not in_string:
            # Aggiungi il record
            fields = current_record.split('\t')
            # Rimuovi gli apici singoli e spazi
            fields = [field.strip().strip('\'') for field in fields]
            records.append(fields)
            current_record = ''
        else:
            current_record += char

    return records


years = [2020, 2021, 2022, 2023]

csv_output_dir = 'dataset/AIS_Dataset_csv'

for year in years:
    # Percorso al file SQL e directory di output CSV
    sql_file_path = f'dataset/AIS_Dataset/ais_stat_data_{year}.sql'

    # Esegue la funzione di conversione con chunk di 40.000 record
    sql_to_csv_chunks(sql_file_path, csv_output_dir, year, chunk_size=40000)

