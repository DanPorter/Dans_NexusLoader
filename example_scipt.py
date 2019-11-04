"""
Example Script Dans_nexusLoader
"""

import Dans_NexusLoader as dnex

exp = dnex.Experiment(r'C:\Users\dgpor\Dropbox\Python\ExamplePeaks', ['E:\I16_Data\mm24570-1'])

print(exp.allscannumbers())

d1 = exp.loadscan(794940)
d2 = exp.loadscan(794935)

dd=d1 + d2
print(dd.info())

d1.plot()

ft = d2.fit(plot_result=True)