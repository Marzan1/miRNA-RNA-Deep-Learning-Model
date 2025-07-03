# A small script to filter the global mature.fa for human sequences.

input_fasta = 'mature.fa'
output_fasta = 'Human_miRNA.fasta'
species_id = 'hsa'  # The 3-letter code for Homo sapiens

print(f"Filtering '{input_fasta}' for species '{species_id}'...")

try:
    with open(input_fasta, 'r') as infile, open(output_fasta, 'w') as outfile:
        write_sequence = False
        count = 0
        for line in infile:
            if line.startswith('>'):
                # Check if the header contains the species ID (e.g., >hsa-let-7a-5p)
                if line.split(' ')[0].startswith('>' + species_id):
                    write_sequence = True
                    outfile.write(line)
                    count += 1
                else:
                    write_sequence = False
            elif write_sequence:
                outfile.write(line)
    print(f"Success! Created '{output_fasta}' with {count} human sequences.")

except FileNotFoundError:
    print(f"Error: Make sure '{input_fasta}' is in the same directory as this script.")