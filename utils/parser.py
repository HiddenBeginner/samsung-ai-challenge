class SDFParser:
    """
    reference
    ----------
    http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx
    """
    @staticmethod
    def parse_counts_line(s):
        num_atoms = int(s[:3])
        s = s[3:]
        
        num_bonds = int(s[:3])

        return (num_atoms, num_bonds)
    
    @staticmethod
    def parse_atoms_block(s):
        # The first three fields, 10 characters long each, describe the atom's position in the X, Y, and Z dimensions.
        coords = [s[10 * i:10 * i + 10] for i in range(3)]
        coords = list(map(lambda x: float(x.strip()), coords))
        s = s[31:]
        
        # Three characters for an atomic symbol
        atom = s[:3].strip()
        s = s[3:]

        # There are two characters for the mass difference from the monoisotope.
        iso = float(s[:2])
        s = s[2:]

        # three characters for the charge.
        chg = float(s[:3])
        s = s[3:]

        # There are ten more fields with three characters each - but these are all rarely used, 
        # and can be left blank for the purposes of working with Progenesis SDF Studio or Progenesis MetaScope.
        remain = [s[3 * i: 3* i + 3] for i in range(10)]
        remain = list(map(lambda x: float(x.strip()), remain))

        return (atom, coords  + [iso] + [chg] + remain)
    
    @staticmethod
    def parse_bonds_block(s):
        u = int(s[:3].strip()) - 1
        s = s[3:]
        
        v = int(s[:3].strip()) - 1
        s = s[3:]
        
        bond_type = int(s[:3].strip())
        s = s[3:]
        
        bond_stereo = int(s[:3].strip())
        
        return (u, v, bond_type, bond_stereo)
        
    
    def parse(self, lines):
        # Parsing the counts line
        count_line = lines[3]
        n_atoms, n_bonds = self.parse_counts_line(count_line)
        
        # Parsing the atoms block
        atoms = lines[4:4 + n_atoms]
        atoms_block = list(map(self.parse_atoms_block, atoms))
        
        # Parsing the bonds block and the properties will be added.
        bonds = lines[4 + n_atoms: 4 + n_atoms + n_bonds]
        bonds_block = list(map(self.parse_bonds_block, bonds))
        # remains = lines[4 + n_atoms + n_bonds:]
                           
        return (n_atoms, n_bonds, atoms_block, bonds_block)