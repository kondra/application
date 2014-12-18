from sys import stdin
from unidecode import unidecode

for line in stdin:
	print unidecode(line)
