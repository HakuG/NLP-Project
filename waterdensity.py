# This is a simple model of the relationship between depth, temperature, and compressibility of
# water. 
# m is the mass
# v is the volume
# k is the bulk modulus


def get_density (m, v):
	return(m/v):
	
def get_speed (m, v, k):
	rho = get_density(m, v)
	return math.sqrt(k/rho)

