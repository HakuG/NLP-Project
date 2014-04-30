# Models the ideal gas laws, 
# pressure, temperature, mass, volume, density

# universal gas constant
R = 287

# Given pressure, temperature, volume, calculate mass and density
def get_mass(p,t,v):
	return((p * v)/(R * t))

def get_density(p,t,v):
	return(p/(R * t))
