
from dataclasses import dataclass
from typing import Union, Sequence, Optional

@dataclass(frozen=True)
class GeneRegion:
    full: str                 # the original string
    gene: str                 # e.g., "TP53"
    segment: str              # e.g., "IDR" / "idr" / "DOMAIN"
    start: int                # e.g., 10
    end: int                  # e.g., 50
    species: Optional[str]    # e.g., "ENSCHYP" or "HOMOSAPIENS" or "HOMO_SAPIENS" or None
    ensembl: Optional[str] #Full ensembl ID e.g. ENSCHYP00000007622
    
    
    @property
    def gene_pos(self) -> str:
        # Reconstructs the token tail (gene + segment + start + end)
        return f"{self.gene}_{self.segment}_{self.start}_{self.end}"

def parse_name(s: str, sep: str = "_") -> GeneRegion:
    """
    Parse strings like:
      HOMO_SAPIENS_TP53_idr_10_50
      TP53_idr_10_50
      ENSCHYP00000007622_Q07002_IDR_1_93
    Strategy: rsplit 3x to get SEGMENT, START, END, and the left prefix.
    Then rsplit the prefix once to get SPECIES and GENE.
    """
    s = s.strip()
    try:
        prefix, segment, start_str, end_str = s.rsplit(sep, 3)
    except ValueError:
        raise ValueError(f"Expected trailing 'SEGMENT{sep}START{sep}END' in: {s}")

    try:
        start = int(start_str)
        end = int(end_str)
    except ValueError:
        raise ValueError(f"Start/end not integers in: {s}")

    # gene = last token before the suffix; species = everything before that (or None)
    if sep in prefix:
        species_full, gene = prefix.rsplit(sep, 1)
        species = re.sub(r"[^A-Za-z]", "", species_full)
        
    else:
        # No species present; entire prefix is the gene
        gene = prefix
        species = None
        species_full=None

    return GeneRegion(full=s, gene=gene, segment=segment, start=start, end=end, species=species,ensembl = species_full)

def get_gene_name_features(
    names: Union[str, Sequence[str]],
    sep: str = "_",
) -> Union[GeneRegion, list[GeneRegion]]:
    if isinstance(names, str):
        return parse_name(names, sep=sep)
    return [parse_name(n, sep=sep) for n in names]


def _parse_ids(self,ortholog_df):
        """
        Vectorized parse of ID strings into __species and __gpos columns.
        Assumes IDs look like 'SPECIES_..._...'; __gpos is the join of the last |pos| tokens.
        """
        pos = self.pos
        id_col = self.id_col
        ids = ortholog_df.iloc[:, id_col].astype(str)
        parts = ids.str.split('_')                      # Series[list[str]]

        k = abs(pos)
        if k == 0:
            raise ValueError("pos must be non-zero; use negative values to take last k tokens")

        # join the last k tokens (vectorized for list-like via .str.slice)
        gpos = parts.str.slice(-k).str.join('_')
        genes = parts.str.get(-k)

        species_full = parts.str.get(0)                 # first token

        # remove non-letters; if result is empty, fall back to original
        species_clean = species_full.str.replace(r'[^A-Za-z]', '', regex=True)
        species = species_clean.where(species_clean.ne(''), species_full)
        
        df = ortholog_df.copy()
        df['__id'] = ids
        df['__species'] = species
        df['__genes'] = genes
        df['__gpos'] = gpos
        return df#, ids.name # return the original id column name too