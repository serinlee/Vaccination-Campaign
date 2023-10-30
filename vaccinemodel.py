#%% Import settings
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import math
import pandas as pd

# %% Initialize

class VaccineModel:
    def get_y0(self):
        SA0 = self.prop_anti * self.prop_sus * self.N_by_group
        SP0 = (1 - self.prop_anti) * self.prop_sus * self.N_by_group
        IA0 = self.prop_anti * self.prop_init_inf * self.N_by_group
        IP0 = (1 - self.prop_anti) * self.prop_init_inf * self.N_by_group
        RA0 = self.prop_anti * (1 - self.prop_sus - self.prop_init_inf) * self.N_by_group
        RP0 = (1 - self.prop_anti) * (1 - self.prop_sus - self.prop_init_inf) * self.N_by_group
        DA0, DP0 = [[0] * len(self.N_by_group)] * 2
        y0 = np.vstack((SA0, IA0, RA0, DA0, SP0, IP0, RP0, DP0)).flatten('F')  # Initial population
        return y0
    
    def get_O_from_physical_contact(self):
        child_rows = np.arange(0, self.num_group, self.num_age_group)
        adult_rows = np.setdiff1d(np.arange(self.num_group), child_rows)
        mask = np.isin(np.arange(self.num_group), adult_rows)[:, None] & np.isin(np.arange(self.num_group), child_rows)
        O = self.C.copy()
        O[mask.T] = 0
        O /= np.sum(O.copy())
        return O
    
    def get_alpha(self):
        return np.array(self.alpha_rr_by_age * self.num_reg_group) * self.overall_alpha

    def get_p(self):
        return (np.array([self.p1,self.p2,self.p3,self.p4,self.p5]*self.num_reg_group))


    def __init__(self, fips_num='53011',init_param_list = [], param_update_list = [], t_f= np.linspace(0, 364, 365), debug=False):
        # Baseline parameters
        self.t_c = np.linspace(0, 58, 59)
        self.calib_period = 59
        self.t_f = t_f
        self.beta = 1.67
        self.O_m = 3.36
        self.k_R = 30
        self.k_E = 30
        self.num_disease = 4
        self.num_vacc_group = 2
        self.num_comp = 8
        self.VE_beta = 0.5
        self.VE_death = 0.1
        self.rho = 1/10
        self.mu = 1/30
        self.alpha_rr_by_age = [0.16, 1.74, 7.83, 25, 100]
        self.overall_alpha = 0.00025
        self.rae = 229
        self.lam = 0.01942
        self.p_online = 0.0
        self.vaccine_risk = 0.0

        # Regional parameters
        self.num_age_group = 5
        self.num_reg_group = 5
        self.num_group = 25
        self.inf_rate_range = np.array([0.04, 0.047, 0.036])
        self.death_rate_range = np.array([1.27, 1.24, 1.32])
        self.vacc_rate_range = np.array([1.0, 0.975, 1.025])
        self.date_label = ["1/1/23", "1/8/23", "1/15/23", "1/22/23", "1/29/23", "2/5/23", "2/12/23", "2/19/23"]
        self.data_date = [i * 7 for i in range(len(self.date_label))]

        # Read data
        self.reg = fips_num
        self.N_by_group = np.loadtxt(f'Data/{self.reg}_pop_by_reg_age.csv', delimiter = ",").flatten()
        self.data_anti_prop = 1-np.loadtxt(f'Data/{self.reg}_vacc_rate_by_time_age_calib.csv', delimiter = ",").T
        self.prop_anti = 1-np.tile(np.loadtxt(f'Data/{self.reg}_vacc_rate_by_time_age.csv', delimiter = ",")[0],self.num_reg_group)            
        self.data_inf_prop = np.loadtxt(f'Data/{self.reg}_data_case.csv', delimiter = ",")/ sum(self.N_by_group)
        self.data_death = np.loadtxt(f'Data/{self.reg}_data_death.csv', delimiter = ",")
        self.C = np.loadtxt(f'Data/{self.reg}_phys_contact_matrix.csv', delimiter = ",")
        self.opinion_online = np.loadtxt(f'Data/{self.reg}_opinion_contact_matrix.csv', delimiter = ",")
        self.opinion_physical =  self.get_O_from_physical_contact()
        self.O = (self.p_online*self.opinion_online + (1-self.p_online)*self.opinion_physical)
        self.prop_sus = 0.71
        self.prop_init_inf = self.data_inf_prop[0] / self.inf_rate_range[0]
        self.p1 = 0.88
        self.p2 = 0.98
        self.p3 = 0.77
        self.p4 = 0.37
        self.p5 = 0.11
        self.alpha = self.get_alpha()
        self.y0 = self.get_y0()
        self.p = self.get_p()
        self.U = [0] * self.num_group
        self.param_update_list = param_update_list
        self.param_updated  = False
        self.debug = debug
        self.init_param_list = init_param_list
        for param_name, param_value in self.init_param_list:
                self.update_param(param_name, param_value)
        
        self.min_rat, self.max_rat,self.min_emo, self.max_emo, self.mean_rat, self.mean_emo = 0,0,0,0,[],[]


    def get_lambda(self, beta, C, I, N):
        eff_N = np.zeros(len(C))
        for i in range(len(C)):
            numerator_sum = sum(C[k][i] * I[k] for k in range(len(C)))
            denominator_sum = sum(C[k][i] * N[k] for k in range(len(C)))
            eff_N[i] = (numerator_sum / denominator_sum)

        lam = np.zeros(len(C))
        for i in range(len(C)):
            for j in range(len(C)):
                lam[i] += C[i][j] * beta[j] * eff_N[j]
        return lam

    def check_dependency(self, param_name):
        if param_name in ["p1","p2","p3","p4","p5"]:
            existing_value = self.p.copy()
            self.p = self.get_p()
            if self.debug == True: print(f"Changed p from {existing_value[:5]} to {self.p[:5]}")

        if param_name in ["alpha_rr_by_age", "overall_alpha"]:
            existing_value = self.alpha.copy()
            self.alpha = self.get_alpha()
            if self.debug == True: print(f"Changed alpha from {existing_value[:5]} to {self.alpha[:5]}")
            
        if param_name == 'p_online':
            existing_value = self.O.copy()
            self.O = (self.p_online*self.opinion_online + (1-self.p_online)*self.opinion_physical)
            if self.debug == True: print(f"Changed O from {existing_value[0][0]} to {self.O[0][0]}")


    def update_param(self, param_name, param_value):
        existing_value = getattr(self, param_name)
        # if type(existing_value) != type(param_value):
        #     raise ValueError(f"Cannot set '{param_name}' with a different data type or format. "
        #                      f"Existing: {existing_value}. New: {param_value}.")
        setattr(self, param_name, param_value)
        if self.debug == True: print(f"Changed {param_name} from {existing_value} to {getattr(self, param_name)}")
        self.check_dependency(param_name)

    def check_range(self, range_rational, range_emotional):
        if(self.min_rat > np.min(range_rational)): self.min_rat = np.min(range_rational)
        if(self.max_rat < np.max(range_rational)): self.max_rat = np.max(range_rational)
        if(self.min_emo > np.min(range_emotional)): self.min_emo = np.min(range_emotional)
        if(self.max_emo < np.max(range_emotional)): self.max_emo = np.max(range_emotional)
        self.mean_rat.append(np.mean(range_rational))
        self.mean_emo.append(np.mean(range_emotional))

    def run_model(self, y, t):
        if (not self.param_updated) and (t > self.t_c[-1]):
            for param_name, param_value in self.param_update_list:
                self.update_param(param_name, param_value)
            self.param_updated = True
    
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(y, (self.num_group,self.num_vacc_group*self.num_disease)))
        A = SA+IA+RA
        P = SP+IP+RP
        N = A+P
        I = IA+IP
        D = DA+DP

        phys_lam=self.get_lambda(self.beta*np.ones(self.num_group), self.C,I,N)
        C_P = -np.divide(self.alpha*self.VE_death * (IP + SP * phys_lam * self.VE_beta), P)- self.vaccine_risk
        C_A = -np.divide(self.alpha*(IA + SA * phys_lam), A)
        R_eta = np.array([1 / (1 + np.exp(-self.k_R * i)) for i in np.subtract(C_P, C_A)])
        R_eta = np.clip(R_eta, 1e-10, 1-1e-10)

        E_eta = np.array([1 / (1 + np.exp(-self.k_E * i)) for i in np.divide(np.subtract(P, A), N)])

        # range_rational = np.subtract(C_P, C_A)
        # range_emotional = np.divide(np.subtract(P, A), N)
        # self.check_range(range_rational, range_emotional)

        eta = (1 - self.p) * R_eta + self.p * E_eta

        opi_AP_lam = np.zeros(self.num_group)
        if t < self.t_c[-1]:
            opi_AP_lam = self.get_lambda(eta*self.O_m, self.O, P*self.lam, N)
        else:
            opi_AP_lam = self.get_lambda(eta*self.O_m, self.O, P*self.lam + self.U, N)
        
        # Changes in opinion (P -> A) and other equations (updated with 'self.')
        dOPi_Sdt =  - opi_AP_lam * SA 
        dOpi_Idt = np.zeros(self.num_group)
        dOpi_Rdt =  - opi_AP_lam * RA

        dSAdt = - SA * phys_lam + dOPi_Sdt + 1 / self.rae * RA
        dSPdt = - self.VE_beta * SP * phys_lam - dOPi_Sdt + 1 / self.rae * RP

        dIAdt = SA * phys_lam - (1 - self.alpha) * self.rho * IA - self.alpha * self.mu * IA + dOpi_Idt
        dIPdt = self.VE_beta * SP * phys_lam - (1 - self.alpha * self.VE_death) * self.rho * IP - self.alpha * self.VE_death * self.mu * IP - dOpi_Idt

        dRAdt = (1 - self.alpha) * self.rho * IA + dOpi_Rdt - 1 / self.rae * RA
        dRPdt = (1 - self.alpha * self.VE_death) * self.rho * IP - dOpi_Rdt - 1 / self.rae * RP

        # Death
        dDAdt = self.alpha * self.mu * IA
        dDPdt = self.alpha * self.VE_death * self.mu * IP

        dydt = [dSAdt, dIAdt, dRAdt, dDAdt, dSPdt, dIPdt, dRPdt, dDPdt]
        return np.array(dydt).ravel(order='F')
    
    def get_results(self, ret):
        sum_death, sum_A, sum_I = 0,0,0
        for l in range(self.num_group):
            SA, IA, RA, DA, SP, IP, RP, DP = np.array(ret.T)[l*self.num_comp:(l+1)*self.num_comp]
            D = np.sum([DA, DP], axis=0)
            sum_death+=(D[-1])
            sum_I += sum(IA+IP)
            sum_A += (SA[-1]+IA[-1]+RA[-1])
        return(sum_death, sum_I, sum_A)


#%%
# from plot import *
# model = VaccineModel()
# ret = odeint(model.run_model, model.get_y0(), model.t_f)
# plot_results_with_calib(model, model.t_f, [ret], error_bar=True)
# %%
