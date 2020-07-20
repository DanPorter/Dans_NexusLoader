"""
Example Script Dans_nexusLoader
"""

import Dans_NexusLoader as dnex

exp = dnex.Experiment([r'C:\Users\dgpor\Dropbox\Python\ExamplePeaks', r'E:\I16_Data\mm24570-1'])

exp.save_config()

print(exp.allscannumbers())

d1 = exp.loadscan(794940)
d2 = exp.loadscan(794935)

dd = d1 + d2
print(dd.info())

d1.plot()

ft = d2.fit(plot_result=True)

print(d1.get_array('/entry1/instrument/pil3_100k/image_data'))

print('----------------------------')

print(d1.find_image_address())

print(d1.filename)

d1.image_plot()

print('Finished')
