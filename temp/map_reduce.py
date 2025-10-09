# DESCRIPTION:
# Парадигма MapReduce.
from functools import reduce

texts = ["регресія кокса аналіз виживаності", "модель пропорційних ризиків регресія кокса"]

def mapper(text):
    words = text.split()
    return [(word.lower(), 1) for word in words]


mapped = [mapper(text) for text in texts]

all_mapped = [item for sublist in mapped for item in sublist]
keys = set([pair[0] for pair in all_mapped])
grouped = {}
for key in keys:
    grouped[key] = []
for pair in all_mapped:
    grouped[pair[0]].append(pair[1])

def reducer(key, values):
    return (key, sum(values))

result = [reducer(key, grouped[key]) for key in grouped]
print(result)
