class EphysRecordingTemplate():
    
    @staticmethod
    def make(table, key):
        sess_dir = find_full_path(get_ephys_root_data_dir(),
                                  get_session_directory(key))
        inserted_probe_serial_number = (ProbeInsertion * probe.Probe & key).fetch1('probe')

        # search session dir and determine acquisition software
        for ephys_pattern, ephys_acq_type in zip(['*.ap.meta', '*.oebin'],
                                                 ['SpikeGLX', 'Open Ephys']):
            ephys_meta_filepaths = list(sess_dir.rglob(ephys_pattern))
            if ephys_meta_filepaths:
                acq_software = ephys_acq_type
                break
        else:
            raise FileNotFoundError(
                f'Ephys recording data not found!'
                f' Neither SpikeGLX nor Open Ephys recording files found')

        supported_probe_types = probe.ProbeType.fetch('probe_type')

        if acq_software == 'SpikeGLX':
            for meta_filepath in ephys_meta_filepaths:
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
                if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                    break
            else:
                raise FileNotFoundError(
                    'No SpikeGLX data found for probe insertion: {}'.format(key))

            if spikeglx_meta.probe_model in supported_probe_types:
                probe_type = spikeglx_meta.probe_model
                electrode_query = probe.ProbeType.Electrode & {'probe_type': probe_type}

                probe_electrodes = {
                    (shank, shank_col, shank_row): key
                    for key, shank, shank_col, shank_row in zip(*electrode_query.fetch(
                        'KEY', 'shank', 'shank_col', 'shank_row'))}

                electrode_group_members = [
                    probe_electrodes[(shank, shank_col, shank_row)]
                    for shank, shank_col, shank_row, _ in spikeglx_meta.shankmap['data']]
            else:
                raise NotImplementedError(
                    'Processing for neuropixels probe model'
                    ' {} not yet implemented'.format(spikeglx_meta.probe_model))

            table.insert1({
                **key,
                **generate_electrode_config(probe_type, electrode_group_members),
                'acq_software': acq_software,
                'sampling_rate': spikeglx_meta.meta['imSampRate'],
                'recording_datetime': spikeglx_meta.recording_time,
                'recording_duration': (spikeglx_meta.recording_duration
                                       or spikeglx.retrieve_recording_duration(meta_filepath))})

            root_dir = find_root_directory(get_ephys_root_data_dir(), meta_filepath)
            table.EphysFile.insert1({
                **key,
                'file_path': meta_filepath.relative_to(root_dir).as_posix()})
        elif acq_software == 'Open Ephys':
            dataset = openephys.OpenEphys(sess_dir)
            for serial_number, probe_data in dataset.probes.items():
                if str(serial_number) == inserted_probe_serial_number:
                    break
            else:
                raise FileNotFoundError(
                    'No Open Ephys data found for probe insertion: {}'.format(key))

            if probe_data.probe_model in supported_probe_types:
                probe_type = probe_data.probe_model
                electrode_query = probe.ProbeType.Electrode & {'probe_type': probe_type}

                probe_electrodes = {key['electrode']: key
                                    for key in electrode_query.fetch('KEY')}

                electrode_group_members = [
                    probe_electrodes[channel_idx]
                    for channel_idx in probe_data.ap_meta['channels_indices']]
            else:
                raise NotImplementedError(
                    'Processing for neuropixels'
                    ' probe model {} not yet implemented'.format(probe_data.probe_model))

            table.insert1({
                **key,
                **generate_electrode_config(probe_type, electrode_group_members),
                'acq_software': acq_software,
                'sampling_rate': probe_data.ap_meta['sample_rate'],
                'recording_datetime': probe_data.recording_info['recording_datetimes'][0],
                'recording_duration': np.sum(probe_data.recording_info['recording_durations'])})

            root_dir = find_root_directory(
                get_ephys_root_data_dir(),
                probe_data.recording_info['recording_files'][0])
            table.EphysFile.insert([{**key,
                                    'file_path': fp.relative_to(root_dir).as_posix()}
                                   for fp in probe_data.recording_info['recording_files']])
        else:
            raise NotImplementedError(f'Processing ephys files from'
                                      f' acquisition software of type {acq_software} is'
                                      f' not yet implemented')


class LFPTemplate():

    @staticmethod
    def make(table, key):
        acq_software = (EphysRecording * ProbeInsertion & key).fetch1('acq_software')

        electrode_keys, lfp = [], []

        if acq_software == 'SpikeGLX':
            spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
            spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)

            lfp_channel_ind = spikeglx_recording.lfmeta.recording_channels[
                          -1::-table._skip_channel_counts]

            # Extract LFP data at specified channels and convert to uV
            lfp = spikeglx_recording.lf_timeseries[:, lfp_channel_ind]  # (sample x channel)
            lfp = (lfp * spikeglx_recording.get_channel_bit_volts('lf')[lfp_channel_ind]).T  # (channel x sample)

            table.insert1(dict(key,
                              lfp_sampling_rate=spikeglx_recording.lfmeta.meta['imSampRate'],
                              lfp_time_stamps=(np.arange(lfp.shape[1])
                                               / spikeglx_recording.lfmeta.meta['imSampRate']),
                              lfp_mean=lfp.mean(axis=0)))

            electrode_query = (probe.ProbeType.Electrode
                               * probe.ElectrodeConfig.Electrode
                               * EphysRecording & key)
            probe_electrodes = {
                (shank, shank_col, shank_row): key
                for key, shank, shank_col, shank_row in zip(*electrode_query.fetch(
                    'KEY', 'shank', 'shank_col', 'shank_row'))}

            for recorded_site in lfp_channel_ind:
                shank, shank_col, shank_row, _ = spikeglx_recording.apmeta.shankmap['data'][recorded_site]
                electrode_keys.append(probe_electrodes[(shank, shank_col, shank_row)])
        elif acq_software == 'Open Ephys':
            oe_probe = get_openephys_probe_data(key)

            lfp_channel_ind = np.r_[
                len(oe_probe.lfp_meta['channels_indices'])-1:0:-table._skip_channel_counts]

            lfp = oe_probe.lfp_timeseries[:, lfp_channel_ind]  # (sample x channel)
            lfp = (lfp * np.array(oe_probe.lfp_meta['channels_gains'])[lfp_channel_ind]).T  # (channel x sample)
            lfp_timestamps = oe_probe.lfp_timestamps

            table.insert1(dict(key,
                              lfp_sampling_rate=oe_probe.lfp_meta['sample_rate'],
                              lfp_time_stamps=lfp_timestamps,
                              lfp_mean=lfp.mean(axis=0)))

            electrode_query = (probe.ProbeType.Electrode
                               * probe.ElectrodeConfig.Electrode
                               * EphysRecording & key)
            probe_electrodes = {key['electrode']: key
                                for key in electrode_query.fetch('KEY')}

            electrode_keys.extend(probe_electrodes[channel_idx]
                                  for channel_idx in oe_probe.lfp_meta['channels_indices'])
        else:
            raise NotImplementedError(f'LFP extraction from acquisition software'
                                      f' of type {acq_software} is not yet implemented')

        # single insert in loop to mitigate potential memory issue
        for electrode_key, lfp_trace in zip(electrode_keys, lfp):
            table.Electrode.insert1({**key, **electrode_key, 'lfp': lfp_trace})


class ClusteringTemplate():

    @staticmethod
    def make(table, key):
        task_mode, output_dir = (ClusteringTask & key).fetch1(
            'task_mode', 'clustering_output_dir')

        if not output_dir:
            output_dir = ClusteringTask.infer_output_dir(key, relative=True, mkdir=True)
            # update clustering_output_dir
            ClusteringTask.update1({**key, 'clustering_output_dir': output_dir.as_posix()})

        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        if task_mode == 'load':
            kilosort.Kilosort(kilosort_dir)  # check if the directory is a valid Kilosort output
        elif task_mode == 'trigger':
            acq_software, clustering_method, params = (ClusteringTask * EphysRecording
                                                       * ClusteringParamSet & key).fetch1(
                'acq_software', 'clustering_method', 'params')

            if 'kilosort' in clustering_method:
                from element_array_ephys.readers import kilosort_triggering

                # add additional probe-recording and channels details into `params`
                params = {**params, **get_recording_channels_details(key)}
                params['fs'] = params['sample_rate']

                if acq_software == 'SpikeGLX':
                    spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
                    spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
                    spikeglx_recording.validate_file('ap')

                    if clustering_method.startswith('pykilosort'):
                        kilosort_triggering.run_pykilosort(
                            continuous_file=spikeglx_recording.root_dir / (
                                        spikeglx_recording.root_name + '.ap.bin'),
                            kilosort_output_directory=kilosort_dir,
                            channel_ind=params.pop('channel_ind'),
                            x_coords=params.pop('x_coords'),
                            y_coords=params.pop('y_coords'),
                            shank_ind=params.pop('shank_ind'),
                            connected=params.pop('connected'),
                            sample_rate=params.pop('sample_rate'),
                            params=params)
                    else:
                        run_kilosort = kilosort_triggering.SGLXKilosortPipeline(
                            npx_input_dir=spikeglx_meta_filepath.parent,
                            ks_output_dir=kilosort_dir,
                            params=params,
                            KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}',
                            run_CatGT=False)
                        run_kilosort.run_modules()
                elif acq_software == 'Open Ephys':
                    oe_probe = get_openephys_probe_data(key)

                    assert len(oe_probe.recording_info['recording_files']) == 1

                    # run kilosort
                    if clustering_method.startswith('pykilosort'):
                        kilosort_triggering.run_pykilosort(
                            continuous_file=pathlib.Path(oe_probe.recording_info['recording_files'][0]) / 'continuous.dat',
                            kilosort_output_directory=kilosort_dir,
                            channel_ind=params.pop('channel_ind'),
                            x_coords=params.pop('x_coords'),
                            y_coords=params.pop('y_coords'),
                            shank_ind=params.pop('shank_ind'),
                            connected=params.pop('connected'),
                            sample_rate=params.pop('sample_rate'),
                            params=params)
                    else:
                        run_kilosort = kilosort_triggering.OpenEphysKilosortPipeline(
                            npx_input_dir=oe_probe.recording_info['recording_files'][0],
                            ks_output_dir=kilosort_dir,
                            params=params,
                            KS2ver=f'{Decimal(clustering_method.replace("kilosort", "")):.1f}')
                        run_kilosort.run_modules()
            else:
                raise NotImplementedError(f'Automatic triggering of {clustering_method}'
                                          f' clustering analysis is not yet supported')

        else:
            raise ValueError(f'Unknown task mode: {task_mode}')

        creation_time, _, _ = kilosort.extract_clustering_info(kilosort_dir)
        table.insert1({**key, 'clustering_time': creation_time})


class CuratedClusteringTemplate():

    @staticmethod
    def make(table, key):
        if table.__module__ == 'ephys_no_curation':
            output_dir = (ClusteringTask & key).fetch1('clustering_output_dir') # set no curation output directory
        else:
            output_dir = (Curation & key).fetch1('curation_output_dir') # set default outpit directory
        
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        kilosort_dataset = kilosort.Kilosort(kilosort_dir)
        acq_software, sample_rate = (EphysRecording & key).fetch1(
            'acq_software', 'sampling_rate')

        sample_rate = kilosort_dataset.data['params'].get('sample_rate', sample_rate)

        # ---------- Unit ----------
        # -- Remove 0-spike units
        withspike_idx = [i for i, u in enumerate(kilosort_dataset.data['cluster_ids'])
                         if (kilosort_dataset.data['spike_clusters'] == u).any()]
        valid_units = kilosort_dataset.data['cluster_ids'][withspike_idx]
        valid_unit_labels = kilosort_dataset.data['cluster_groups'][withspike_idx]
        # -- Get channel and electrode-site mapping
        channel2electrodes = get_neuropixels_channel2electrode_map(key, acq_software)

        # -- Spike-times --
        # spike_times_sec_adj > spike_times_sec > spike_times
        spike_time_key = ('spike_times_sec_adj' if 'spike_times_sec_adj' in kilosort_dataset.data
                          else 'spike_times_sec' if 'spike_times_sec'
                                                    in kilosort_dataset.data else 'spike_times')
        spike_times = kilosort_dataset.data[spike_time_key]
        kilosort_dataset.extract_spike_depths()

        # -- Spike-sites and Spike-depths --
        spike_sites = np.array([channel2electrodes[s]['electrode']
                                for s in kilosort_dataset.data['spike_sites']])
        spike_depths = kilosort_dataset.data['spike_depths']

        # -- Insert unit, label, peak-chn
        units = []
        for unit, unit_lbl in zip(valid_units, valid_unit_labels):
            if (kilosort_dataset.data['spike_clusters'] == unit).any():
                unit_channel, _ = kilosort_dataset.get_best_channel(unit)
                unit_spike_times = (spike_times[kilosort_dataset.data['spike_clusters'] == unit]
                                    / sample_rate)
                spike_count = len(unit_spike_times)

                units.append({
                    'unit': unit,
                    'cluster_quality_label': unit_lbl,
                    **channel2electrodes[unit_channel],
                    'spike_times': unit_spike_times,
                    'spike_count': spike_count,
                    'spike_sites': spike_sites[kilosort_dataset.data['spike_clusters'] == unit],
                    'spike_depths': spike_depths[kilosort_dataset.data['spike_clusters'] == unit]})

        table.insert1(key)
        table.Unit.insert([{**key, **u} for u in units])


class WaveformSetTemplate():

    @staticmethod
    def make(table, key):
        if table.__module__ == 'ephys_no_curation':
            output_dir = (ClusteringTask & key).fetch1('clustering_output_dir') # set no curation output directory
        else:
            output_dir = (Curation & key).fetch1('curation_output_dir') # set default outpit directory
            
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        kilosort_dataset = kilosort.Kilosort(kilosort_dir)

        acq_software, probe_serial_number = (EphysRecording * ProbeInsertion & key).fetch1(
            'acq_software', 'probe')

        # -- Get channel and electrode-site mapping
        recording_key = (EphysRecording & key).fetch1('KEY')
        channel2electrodes = get_neuropixels_channel2electrode_map(recording_key, acq_software)

        # Get all units
        units = {u['unit']: u for u in (CuratedClustering.Unit & key).fetch(
            as_dict=True, order_by='unit')}

        if table.__module__ == 'ephys_no_curation':
            unit_waveforms_condition = (kilosort_dir / 'mean_waveforms.npy').exists() # set ephys_no_curation conditional
        else:
            unit_waveforms_condition = (Curation & key).fetch1('quality_control') # set default conditional

        if unit_waveforms_condition:
            unit_waveforms = np.load(kilosort_dir / 'mean_waveforms.npy')  # unit x channel x sample

            def yield_unit_waveforms():
                for unit_no, unit_waveform in zip(kilosort_dataset.data['cluster_ids'],
                                                  unit_waveforms):
                    unit_peak_waveform = {}
                    unit_electrode_waveforms = []
                    if unit_no in units:
                        for channel, channel_waveform in zip(
                                kilosort_dataset.data['channel_map'],
                                unit_waveform):
                            unit_electrode_waveforms.append({
                                **units[unit_no], **channel2electrodes[channel],
                                'waveform_mean': channel_waveform})
                            if channel2electrodes[channel]['electrode'] == units[unit_no]['electrode']:
                                unit_peak_waveform = {
                                    **units[unit_no],
                                    'peak_electrode_waveform': channel_waveform}
                    yield unit_peak_waveform, unit_electrode_waveforms
        else:
            if acq_software == 'SpikeGLX':
                spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
                neuropixels_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
            elif acq_software == 'Open Ephys':
                sess_dir = find_full_path(get_ephys_root_data_dir(),
                                          get_session_directory(key))
                openephys_dataset = openephys.OpenEphys(sess_dir)
                neuropixels_recording = openephys_dataset.probes[probe_serial_number]

            def yield_unit_waveforms():
                for unit_dict in units.values():
                    unit_peak_waveform = {}
                    unit_electrode_waveforms = []

                    spikes = unit_dict['spike_times']
                    waveforms = neuropixels_recording.extract_spike_waveforms(
                        spikes, kilosort_dataset.data['channel_map'])  # (sample x channel x spike)
                    waveforms = waveforms.transpose((1, 2, 0))  # (channel x spike x sample)
                    for channel, channel_waveform in zip(
                            kilosort_dataset.data['channel_map'], waveforms):
                        unit_electrode_waveforms.append({
                            **unit_dict, **channel2electrodes[channel],
                            'waveform_mean': channel_waveform.mean(axis=0),
                            'waveforms': channel_waveform})
                        if channel2electrodes[channel]['electrode'] == unit_dict['electrode']:
                            unit_peak_waveform = {
                                **unit_dict,
                                'peak_electrode_waveform': channel_waveform.mean(axis=0)}

                    yield unit_peak_waveform, unit_electrode_waveforms

        # insert waveform on a per-unit basis to mitigate potential memory issue
        table.insert1(key)
        for unit_peak_waveform, unit_electrode_waveforms in yield_unit_waveforms():
            if unit_peak_waveform:
                table.PeakWaveform.insert1(unit_peak_waveform, ignore_extra_fields=True)
            if unit_electrode_waveforms:
                table.Waveform.insert(unit_electrode_waveforms, ignore_extra_fields=True)
