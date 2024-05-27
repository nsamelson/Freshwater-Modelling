import pandas as pd
import xlwings as xw
import json


def main():
    # Read sheets
    calcs_sheet = xw.Book("data/WBal Calcs.xlsm").sheets['Calcs']
    params_sheet = xw.Book("data/WBal Calcs.xlsm").sheets['Param']

    # Read and setup parameters
    params = get_params(params_sheet)
    D_prof_max = params.get("D_prof_max",0)
    E_max = params.get("E_max",0)
    D_surf_max = params.get("D_surf_max",0)
    E_surf_split = params.get("E_surf_split",0)
    max_rows = params.get("max_rows",0)
    rain_col = params.get("rain_col",0)
    # et_col = params.get("et_col",0)
    # def_col = params.get("def_col",0)
    pet_col = params.get("pet_col",0)

    # initialise other variables
    D_surf,D_prof,E_act = 0,0,0

    # get calcs sheet into 2D list
    calcs = calcs_sheet.range(f"A2:J{int(max_rows)+1}").value  
    headers = calcs_sheet.range("A1:J1").value
    print("HEADERS: ",headers)

    results = {
        "D_prof":[],
        "E_act":[],
        "D_surf":[],
        "Drain":[],
    } 

    for i,row in enumerate(calcs):
        if row[3] >= 0.1: # New year thing
            D_surf = 0
            D_prof = 0

        m_rain = row[int(rain_col) - 1]
        m_pet = row[int(pet_col) - 1]

        D_prof, E_act, D_surf, drain = run_one_day(D_surf,D_prof,E_act,E_surf_split, m_rain, m_pet, D_surf_max, D_prof_max, E_max)
        results["D_prof"].append(D_prof)
        results["E_act"].append(E_act)
        results["D_surf"].append(D_surf)
        results["Drain"].append(drain)
    
    save_results(results)

def save_results(results):
    df = pd.DataFrame(results)
    df.to_csv('data/BalResults.csv', index=False)  

def run_one_day(D_surf,D_prof,E_act,E_surf_split, m_rain, m_pet, D_surf_max, D_prof_max, E_max):

    # Initial update of deficits
    D_surf += m_rain
    D_prof += m_rain

    # what could the actual evaporation from surface and profile be?
    E_surf = min(m_pet, max(D_surf - D_surf_max, 0)) # if from the surface then is the min of pet and the surface storage, don't let the surface storage be -ve
    E_prof = min(m_pet, E_max - min(0,D_prof) / D_prof_max * E_max) # if from the profile then is the minimum of pet and the deficit-controlled evaporation rate

    if E_surf >= E_prof: # evaporation is surface controlled
        E_act = E_surf
        E_prof = 0
    else: # evaporation is profile controlled
        E_act = E_prof
        E_surf = 0

    # Update and limit the deficits
    D_surf -= E_act
    D_surf = min(0,max(D_surf_max, D_surf)) # limit the range

    D_prof -= E_act
    drain = max(0,D_prof)
    D_prof = min(0,max(D_prof_max, D_prof))

    return D_prof, E_act, D_surf, drain 

def get_params(params_sheet):
    param_names = params_sheet.range("A4:A14").value
    value_names = params_sheet.range("B4:B14").value

    params =  {}
    
    for i,param in enumerate(param_names):
        if param == "blank" or param == None:
            continue
        params[param] = value_names[i]
    return params
    
if __name__ == "__main__":
    main()