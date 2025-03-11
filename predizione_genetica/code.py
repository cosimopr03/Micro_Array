
import pandas as pd
import regex
import Bio
import Bio.SeqIO
from bisect import bisect_left
import wget


class Match:
	def __init__(self, string, start, stop):
		self.string = string
		self.start_pos = start
		self.stop_pos = stop
		self._group0 = string[start:stop]
	def group(self):
		return self._group0
	def span(self):
		return (self.start_pos, self.stop_pos)
	def __repr__(self):
		return f"<Match object; span={self.span()}, match='{self.group()}'>"

def orf_iter(sequence):
	# Cerco tutti i codoni di stop e salvo il punto di partenza
	stops = []
	for stop in regex.finditer(r"TAA|TAG|TGA", sequence):
		stops.append(stop.span()[0])
	# Itero sugli start
	for start in regex.finditer(r"ATG", sequence):
		# cerco il primo stop con valore di distanza multiplo di 3
		pos_start = start.span()[0]
		pos_stop = bisect_left(stops, pos_start + 3)
		while (pos_stop < len(stops)) and ((stops[pos_stop] - pos_start) % 3 != 0):
			pos_stop = pos_stop + 1
		if pos_stop < len(stops):
			yield Match(sequence, pos_start, stops[pos_stop])

def contenuto_GC(orf):
	numG=orf.count('G')
	numC= orf.count('C')
	ret=((numG+numC)/len(orf))
	return ret>0.40 and ret<0.65

	
def Iniziatore(sequenza,  inizio_orf):
	regIniziatore="[CT][CT]A[ACGT][AT][CT][CT]"
	## CERCHIAMO L'INIZIATORE
	indice_inizio=inizio_orf-200
	regione_da_analizzare=sequenza[indice_inizio: inizio_orf]
	if(regex.search(regIniziatore, regione_da_analizzare)!=None):
		return regex.search(regIniziatore, regione_da_analizzare).span()[0]
	return -1

def TataBox(sequenza, indice_iniziatore):
	regTATA= "TATA[AT]{1}A[AT]{1}"
	## CERCHIAMO LA TATABOX
	regione_da_analizzare=sequenza[indice_iniziatore-40: indice_iniziatore+1]
	if(regex.search(regTATA, regione_da_analizzare)!=None):
		return True
	return False

def sequenza_kozak(sequenza, inizio_orf):
	regKozak="(GCC)?GCC[AG]CCATGG"
	indice_inizio=inizio_orf-10
	indice_fine=inizio_orf+4
	regione= sequenza[indice_inizio:indice_fine+1]
	if(regex.search(regKozak, regione)!=None):
		return True
	return False


def iniziatore_tata_box(sequenza, inizio_orf):
	regIniziatore="[CT][CT]A[ACGT][AT][CT][CT]"
	indice_inizio= max (0,inizio_orf-400)
	indice_iniziatore=0
	regione_da_analizzare=sequenza[indice_inizio: inizio_orf]
	if(regex.search(regIniziatore, regione_da_analizzare)!=None):
		indice_iniziatore=regex.search(regIniziatore, regione_da_analizzare).span()[0]+indice_inizio
	else:
		return False
	regTATA= "TATA[AT]A[AT]"
	indice_seconda=max(0,indice_iniziatore-50)
	regione2= sequenza[indice_seconda: indice_iniziatore+1]
	if(regex.search(regTATA, regione2)!=None):
		return True
	return False

def isole_cpg(seq,inidice_inizio_orf):
	indice_cpg=max(inidice_inizio_orf-3000,0)
	for i in range(indice_cpg,indice_cpg+3000):
		regione_isola=seq[i:i +300]
		numG=regione_isola.count('G')
		numC= regione_isola.count('C')
		contenuto_gc=((numG+numC)/len(regione_isola)) 
		rapporto_osservato=regione_isola.count("CG")
		rapporto_atteso=numC*numG
		rapporto=  (rapporto_osservato / rapporto_atteso)* len(regione_isola)
		if rapporto>0 and contenuto_gc>0 :
			return True
		i+=1
	return False

def prog(seq,path):	
	# Lettura da file di testo
	results = pd.DataFrame(columns=["ORFpos","Coding","BLAST"])
	for orf in orf_iter(str(seq)):		
		##filtrare lunghezza-->almeno 150
		lunghezza= ( orf.span()[1]- orf.span()[0])
		if( lunghezza>300):
			results.loc[len(results)] = [orf.span()[0], False, False]
			if(  iniziatore_tata_box(str(seq), orf.span()[0])  and contenuto_GC(str(orf)) and sequenza_kozak(str(seq), orf.span()[0]) and  isole_cpg(str(seq),orf.span()[0] )  ):
				results.loc[len(results)-1,"Coding"] = True	
				print( "\n", orf)
	results.to_csv(path)
		

def main():
	url= '  '
	fileName=wget.download(url)
	seq1 = next(Bio.SeqIO.parse(fileName, "fasta")).seq
	seq2=next(Bio.SeqIO.parse(fileName, "fasta")).reverse_complement().seq #sequenza inversa
	path_1="percorso_1"
	path_2="percorso_2"
	prog(seq1,path_1)
	prog(seq2,path_2)

main()
