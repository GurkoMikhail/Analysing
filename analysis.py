# from subjects import Detector
import numpy as np
import h5py


def modeling_parameters(file_name):
    print('Start parameters')
    file_in = h5py.File(f'Raw data/{file_name}', 'a')
    file_out = h5py.File(f'Processed data/{file_name}', 'a')
    file_in['Modeling parameters'].copy(file_in['Modeling parameters'], file_out)
    file_in.close()
    file_out.close()
    print('Finish parameters')

def inside_subject(file_name, subject):
    print('Start inside')
    coordinates = []
    energy_transfer = []
    emission_time = []
    emission_coordinates = []
    file_in = h5py.File(f'Raw data/{file_name}', 'a')
    for flow in file_in['Flows'].values():
        _coordinates = np.copy(flow['Coordinates'])
        _energy_transfer = np.copy(flow['Energy transfer'])
        _emission_time = np.copy(flow['Emission time'])
        _emission_coordinates = np.copy(flow['Emission coordinates'])
        indices = subject.inside(_coordinates)
        coordinates.append(_coordinates[indices])
        energy_transfer.append(_energy_transfer[indices])
        emission_time.append(_emission_time[indices])
        emission_coordinates.append(_emission_coordinates[indices])
    coordinates = np.concatenate(coordinates)
    energy_transfer = np.concatenate(energy_transfer)
    emission_time = np.concatenate(emission_time)
    emission_coordinates = np.concatenate(emission_coordinates)
    file_in.close()
    file_out = h5py.File(f'Processed data/{file_name}', 'a')
    in_subject = file_out.create_group(f'Inside_{subject.__class__.__name__}')
    in_subject.create_dataset('Coordinates', data=coordinates)
    in_subject.create_dataset('Energy transfer', data=energy_transfer)
    in_subject.create_dataset('Emission time', data=emission_time)
    in_subject.create_dataset('Subject size', data=subject.size)
    in_subject.create_dataset('Emission coordinates', data=emission_coordinates)
    file_out.close()
    print('Finish inside')


def dose_distribution(file_name, space_size, voxel_size):
    print('Start dose')
    volume = np.zeros((space_size/voxel_size).astype(np.uint))
    file_in = h5py.File(f'Raw data/{file_name}', 'a')
    for flow in file_in['Flows'].values():
        coordinates = flow['Coordinates']
        energy_transfer = flow['Energy transfer']
        flow_volume = np.histogramdd(
            sample=coordinates,
            bins=(space_size/voxel_size).astype(np.int),
            range=((0, space_size[0]), (0, space_size[1]), (0, space_size[2])),
            weights=energy_transfer
        )[0]
        volume += flow_volume
    file_in.close()
    file_out = h5py.File(f'Processed data/{file_name}', 'a')
    group = file_out.create_group('Dose distribution')
    group.create_dataset('Volume', data=volume)
    group.create_dataset('Voxel size', data=voxel_size)
    file_out.close()
    print('Finish dose')


def emission_distribution(file_name, space_size, voxel_size):
    print('Start emission')
    volume = np.zeros((space_size/voxel_size).astype(np.uint))
    file_in = h5py.File(f'Raw data/{file_name}', 'a')
    for flow in file_in['Flows'].values():
        emission_time = np.asarray(flow['Emission time'])
        unique, indices = np.unique(emission_time, return_index=True)
        coordinates = np.asarray(flow['Emission coordinates'])[indices]
        energy_transfer = np.asarray(flow['Energy transfer'])[indices]
        flow_volume = np.histogramdd(
            sample=coordinates,
            bins=(space_size/voxel_size).astype(np.int),
            range=((0, space_size[0]), (0, space_size[1]), (0, space_size[2])),
            weights=energy_transfer
        )[0]
        volume += flow_volume
    file_in.close()
    file_out = h5py.File(f'Processed data/{file_name}', 'a')
    group = file_out.create_group('Emission distribution')
    group.create_dataset('Volume', data=volume)
    group.create_dataset('Voxel size', data=voxel_size)
    file_out.close()
    print('Finish emission')


def source_distribution(file_name, voxel_size):
    print('Start source')
    file_in = h5py.File(f'Raw data/{file_name}', 'a')
    volume = np.copy(file_in['Source distribution'])
    file_in.close()
    file_out = h5py.File(f'Processed data/{file_name}', 'a')
    group = file_out.create_group('Source distribution')
    group.create_dataset('Volume', data=volume)
    group.create_dataset('Voxel size', data=voxel_size)
    file_out.close()
    print('Finish source')

def concatenate_flows(file_name):
    coordinates = []
    energy_transfer = []
    emission_time = []
    emission_coordinates = []
    file = h5py.File(f'Raw data/{file_name}', 'a')
    subject_name = 'Detector'
    subject_size = np.copy(file[f'Modeling parameters/Space/{subject_name}/size'])
    for flow in file['Flows'].values():
        coordinates.append(np.copy(flow['Coordinates']))
        energy_transfer.append(np.copy(flow['Energy transfer']))
        emission_time.append(np.copy(flow['Emission time']))
        emission_coordinates.append(np.copy(flow['Emission coordinates']))
    coordinates = np.concatenate(coordinates)
    energy_transfer = np.concatenate(energy_transfer)
    emission_time = np.concatenate(emission_time)
    emission_coordinates = np.concatenate(emission_coordinates)
    in_subject = file.create_group(f'Inside {subject_name}')
    in_subject.create_dataset('Coordinates', data=coordinates)
    in_subject.create_dataset('Energy transfer', data=energy_transfer)
    in_subject.create_dataset('Emission time', data=emission_time)
    in_subject.create_dataset('Subject size', data=subject_size)
    in_subject.create_dataset('Emission coordinates', data=emission_coordinates)
    # del file['Flows']
    file.close()

if __name__ == '__main__':
    file_name = 'efg3_fix 0.0 deg.hdf'
    space_size = np.asarray((53.3, 70., 38.7))
    voxel_size = 0.4
    # subject = Detector(
    #     coordinates=(0., 9.5, 0.),
    #     size=(53.3, 38.7, 9.5),
    #     material=4,
    #     euler_angles=(0, np.pi/2, 0),
    #     rotation_center=(0., 0., 0.)
    #     )
    
    concatenate_flows(file_name)

    # modeling_parameters(file_name)
    # source_distribution(file_name, voxel_size)
    # dose_distribution(file_name, space_size, voxel_size)
    # emission_distribution(file_name, space_size, voxel_size)
    # inside_subject(file_name, subject)

