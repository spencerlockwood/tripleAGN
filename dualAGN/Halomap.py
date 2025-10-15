import tangos
import pynbody as pnb
import numpy as np
import h5py
import argparse as ap

####################################################################################################
parser = ap.ArgumentParser()
parser.add_argument('DATA_FOLDER')
parser.add_argument('SLURM_ARRAY_TASK_ID')
args = parser.parse_args()

snapshots = ['000098','000105','000111','000118','000126','000134','000142','000151','000160','000170','000181',\
            '000192','000204','000216','000229','000243','000256','000258','000274','000290','000308',\
            '000372','000437','000446',\
            '000463','000491','000512','000520','000547','000551','000584','000618','000655',\
            '000690','000694','000735','000768','000778','000824','000873','000909','000924','000979','001024',\
            '001036','001065','001097','001162','001230','001270','001280',\
            '001302','001378','001458','001536','001543','001550','001632','001726','001792','001826','001931','001945',\
            '002042','002048','002159','002281','002304','002411','002536','002547','002560','002690','002816','002840',\
            '002998','003072','003163','003328','003336','003478','003517','003584','003707','003840','003905','004096',\
            '004111','004173','004326','004352','004549','004608','004781','004864','005022','005107','005120','005271',\
            '005376','005529','005632','005795','005888','006069','006144','006350','006390','006400','006640','006656',\
            '006912','006937','007168','007212','007241','007394','007424','007552','007680','007779','007869','007936',\
            '008192']

a = '%s' % (args.SLURM_ARRAY_TASK_ID)
ts = int(a)
snap = snapshots[ts]

print('snap',snap)
print('timestep',ts)

####################################################################################################
sim = tangos.get_simulation("cosmo25")
timestep = sim.timesteps[ts]
allBHs = timestep.bhs
allBHs = list(allBHs)
# halos_query = sim.timesteps[ts].halos
# halos_list = list(halos_query)
# num_halos = len(halos_list)

####################################################################################################
bh_id = np.zeros(len(allBHs))
bh_mdot = np.zeros(len(allBHs))
bh_mass = np.zeros(len(allBHs))
bh_pnb_haloid = np.zeros(len(allBHs))
bh_tng_haloid = np.zeros(len(allBHs))

for j in range(len(allBHs)):
    print(j)
    try:
        bh_id[j] = allBHs[j].finder_id
        bh_mdot[j] = allBHs[j]['BH_mdot']
        bh_mass[j] = allBHs[j]['BH_mass']
        bh_pnb_haloid[j] = allBHs[j]['host_halo'].finder_id
        bh_tng_haloid[j] = allBHs[j]['host_halo'].halo_number
    except:
        print('No BH info')
        pass

####################################################################################################
#hf = h5py.File('/home/vidasz/projects/rrg-babul-ad/vidasz/DualAGN/Datasets/CtalougueTest-HaloBH-TangosPynbodyMap-R25-snap{0}.hdf5'.format(snap), 'w')
hf = h5py.File('/home/vidasz/scratch/DualAGNs/Datasets/HaloBH-TangosPynbodyMap-R25-snap{0}.hdf5'.format(snap), 'w')

hf.create_dataset('pynbody_haloid', data=bh_pnb_haloid)
hf.create_dataset('tangos_haloid', data=bh_tng_haloid)

hf.create_dataset('BH_id', data=bh_id)
hf.create_dataset('BH_mdot', data=bh_mdot)
hf.create_dataset('BH_mass', data=bh_mass)

hf.close()
