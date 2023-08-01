"""
As update_elkan just calls smaller, unit-testable functions it 
is tested in the KKMeans test folder as more of an integration 
test. There it is run against the vanilla lloyd's kernel kmeans
and (with linear kernel) against sklearn KMeans (with proper seeding). 
"""
