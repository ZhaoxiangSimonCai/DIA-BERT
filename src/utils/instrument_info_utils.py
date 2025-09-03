import numpy as np

from src.common.model.score_model import FeatureEngineer


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_mzml_nearest_rt(mzml_rt):
    fe = FeatureEngineer()
    rt_list = list(fe.rt_s2i.keys())
    mzml_rt = find_nearest(rt_list, mzml_rt)
    return mzml_rt


def get_mzml_meta_info(filepath):
    with open(filepath, 'r') as file:  ###LZW20241226
        lines = file.readlines()

    # filter lines with useful information
    lines_sub = lines[:500]
    MS2_id = [i for i in range(len(lines_sub)) if 'name="ms level" value="2"' in lines_sub[i]]
    if len(MS2_id) < 2:
        lines_sub = lines[:5000]
        MS2_id = [i for i in range(len(lines_sub)) if 'name="ms level" value="2"' in lines_sub[i]]
        if len(MS2_id) < 2:
            lines_sub = lines[:100000]
            MS2_id = [i for i in range(len(lines_sub)) if 'name="ms level" value="2"' in lines_sub[i]]
            if len(MS2_id) < 2:  ######### LZW20241206
                lines_sub = lines  ######### LZW20241206
                MS2_id = [i for i in range(len(lines_sub)) if
                          'name="ms level" value="2"' in lines_sub[i]]  ######### LZW20241206
                if len(MS2_id) < 2:
                    raise ValueError(f'Not enough MS2 found in {filepath} !!!')
    lines_sub = lines_sub[:MS2_id[1]]

    # auto detect instrument type
    instrument_id = [i for i in range(len(lines_sub)) if
                     '<referenceableParamGroup id="CommonInstrumentParams">' in lines_sub[i]]
    if len(instrument_id) == 1:
        instrument = lines_sub[instrument_id[0] + 1].split('name="')[-1].split('"')[0]
    else:
        if len(instrument_id) > 1:
            instrument = lines_sub[instrument_id[0] + 1].split('name="')[-1].split('"')[0]
            print(f'Please check the instrument information! \n {np.array(lines_sub)[[_ + 1 for _ in instrument_id]]}',
                  flush=True)
        else:
            instrument_id = [i for i in range(len(lines_sub)) if '<instrumentConfiguration id=' in lines_sub[i]]
            if len(instrument_id) == 1:
                instrument = lines_sub[instrument_id[0] + 1].split('name="')[-1].split('"')[0]
            else:
                if len(instrument_id) > 1:
                    instrument = lines_sub[instrument_id[0] + 1].split('name="')[-1].split('"')[0]
                    print(
                        f'Please check the instrument information! \n {np.array(lines_sub)[[_ + 1 for _ in instrument_id]]}',
                        flush=True)
                else:
                    raise ValueError(f'No instrument information found !!!')

    # auto detect RT_unit
    RT_unit_T = [_ for _ in lines_sub if 'scan start time' in _]
    if len(RT_unit_T) > 0:
        RT_unit = RT_unit_T[0].split('unitName="')[-1].split('"')[0]
    else:
        raise Exception("RT_unit can't be detected !")
    return instrument, RT_unit
