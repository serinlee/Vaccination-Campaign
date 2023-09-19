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
    
    # def get_O(self):
    #     child_rows = np.arange(0, self.num_group, self.num_age_group)
    #     adult_rows = np.setdiff1d(np.arange(self.num_group), child_rows)
    #     mask = np.isin(np.arange(self.num_group), adult_rows)[:, None] & np.isin(np.arange(self.num_group), child_rows)
    #     O = self.C / self.O_m
    #     O[mask] = 0
    #     return O
    
    def get_alpha(self):
        return np.array(self.alpha_rr_by_age * self.num_reg_group) * self.overall_alpha

    def get_p(self):
        return (np.array([self.p1,self.p2,self.p3,self.p4,self.p5]*self.num_reg_group))


    def __init__(self, fips_num='53011', param_update_list = [], param_update_mode = None, t_f= np.linspace(0, 364, 365), debug=False):
        # Baseline parameters
        self.t_c = np.linspace(0, 58, 59)
        self.calib_period = 59
        self.t_f = t_f
        self.beta = 1.67
        self.O_m = 3.36
        self.k_R = 346019
        self.k_E = 0.81
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
        self.O = self.O_m*np.loadtxt(f'Data/{self.reg}_opinion_contact_matrix.csv', delimiter = ",")

        # self.N_by_group = [17239., 14081., 12210., 13284., 9836., 15480., 9186., 13548., 11921., 8124., 9972., 5596., 7076., 7443., 5851., 17463., 16710., 14370., 12228., 8448., 54748., 54926., 47929., 46679., 38904.]
        # self.prop_anti = (1 - np.array([0.0752771242289099, 0.112871899463411, 0.169634068557727, 0.224438921476377, 0.533460778527623] * self.num_reg_group))   
        # self.data_anti_prop =  (1 - np.array([
                #     [0.0752771242289099, 0.0779342393276445, 0.0805445330149477, 0.0824993269422107, 0.0839507906965856, 0.0855895400966862, 0.0866781379124674, 0.0883754140768574, ],
                #     [0.167242306664457, 0.171183270018572, 0.174613430397175, 0.177228335695656, 0.179481281269768, 0.181565086785764, 0.183364737004124, 0.185117028006211],
                #     [0.533460778527623, 0.541158810397475, 0.547650817274383, 0.553154910061328, 0.556978265890021, 0.560660491134433, 0.564381206538195, 0.567640040029766]
                # ]))
        # self.data_inf_prop = (np.array([376, 347, 335, 325, 331, 327, 309, 273]) / sum(self.N_by_group))
        # self.data_death = np.array([1, 15, 21, 25, 33, 45, 48, 54])
        # self.C =  np.array(pd.read_csv('Data/contact_matrix_' + self.reg + '.csv', header=None))
        # self.O = self.get_O()

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
        self.param_update_mode = param_update_mode
        self.debug = debug

    def get_lambda(self, C, I, N):
        lam=[0]*len(C)
        for i in range(len(C)):
            for j in range(len(C)):
                lam[i]+=C[i][j]*I[j]/N[j]
            return lam

    # def get_lambda(self, C, I, N):
    #     lam = [0] * len(C)
    #     for i in range(len(C)):
    #         for j in range(len(C)):
    #             numerator_sum = sum(C[k][j] * I[k] for k in range(len(C)))
    #             denominator_sum = sum(C[k][j] * N[k] for k in range(len(C)))
    #             lam[i] += C[i][j] * (numerator_sum / denominator_sum)
    #     return lam

    def check_dependency(self, param_name):
        if param_name in ["p1","p2","p3","p4","p5"]:
            existing_value = self.p
            self.p = self.get_p()
            if self.debug == True: print(f"Changed p from {existing_value[:5]} to {self.p[:5]}")

        if param_name == 'O_m':
            existing_value = self.O
            self.O *= self.O_m
            if self.debug == True: print(f"Changed O from .{existing_value[0][0]:.4f} to {self.O[0][0]:.4f}")

        if param_name in ["alpha_rr_by_age", "overall_alpha"]:
            existing_value = self.alpha
            self.alpha = self.get_alpha()
            if self.debug == True: print(f"Changed alpha from {existing_value[:5]} to {self.alpha[:5]}")

    def update_param(self, param_name, param_value):
        existing_value = getattr(self, param_name)
        # if type(existing_value) != type(param_value):
        #     raise ValueError(f"Cannot set '{param_name}' with a different data type or format. "
        #                      f"Existing: {existing_value}. New: {param_value}.")
        setattr(self, param_name, param_value)
        if self.debug == True: print(f"Changed {param_name} from {existing_value} to {getattr(self, param_name)}")
        self.check_dependency(param_name)

    def run_model(self, y, t):
        if not self.param_updated:
            if (self.param_update_mode is None and t > self.t_c[-1]) or self.param_update_mode == 'Calibration':
                for param_name, param_value in self.param_update_list:
                    self.update_param(param_name, param_value)
                self.param_updated = True
    
        [SA, IA, RA, DA, SP, IP, RP, DP] = np.transpose(np.reshape(y, (self.num_group,self.num_vacc_group*self.num_disease)))
        A = SA+IA+RA
        P = SP+IP+RP
        N = A+P
        I = IA+IP
        D = DA+DP

        prop_anti = np.divide(A,N)
        phys_lam=self.get_lambda(self.C,I,N)
        C_P = -np.divide(self.alpha * self.mu * self.VE_death * (IP + self.beta * SP * phys_lam * self.VE_beta), P)
        C_A = -np.divide(self.alpha * self.mu * (IA + self.beta * SA * phys_lam), A)
        max_exp_arg = 709.7827
        R_eta = np.array([1 / (1 + np.exp(np.clip(-self.k_R * i, -max_exp_arg, max_exp_arg))) for i in np.subtract(C_P, C_A)])  # clip to avoid error
        R_eta = np.clip(R_eta, 0.00001, 0.99999)

        # Emotional Judegement
        P_tilde = self.get_lambda(self.O, P, [1] * self.num_group)
        A_tilde = self.get_lambda(self.O, A, [1] * self.num_group)
        N_tilde = np.sum([P_tilde, A_tilde], axis=0)
        E_eta = np.array([1 / (1 + np.exp(-self.k_E * (i))) for i in np.divide(np.subtract(P_tilde, A_tilde), N_tilde)])
        E_eta = np.clip(E_eta, 0.00001, 0.99999)

        p_eta = (1 - self.p) * R_eta + self.p * E_eta

        eta = np.array([-math.log(1 - i) for i in p_eta])

        # Update the equation. Intervention takes place here
        opi_AP_lam = [0] * self.num_group
        for i in range(self.num_group):
            for j in range(self.num_group):
                if t < self.t_c[-1]: 
                    opi_AP_lam[i] += self.O[i][j] * (1 - self.prop_anti[j]) * (self.lam) 
                else:
                    opi_AP_lam[i] += self.O[i][j] * (1 - self.prop_anti[j]) * (self.lam + self.U[j] / P[j])

        # Changes in opinion (P -> A) and other equations (updated with 'self.')
        dOPi_Sdt =  - eta * SA * opi_AP_lam
        dOpi_Idt = np.zeros(self.num_group)
        dOpi_Rdt =  - eta * RA * opi_AP_lam

        dSAdt = -self.beta * SA * phys_lam + dOPi_Sdt + 1 / self.rae * RA
        dSPdt = -self.beta * self.VE_beta * SP * phys_lam - dOPi_Sdt + 1 / self.rae * RP

        dIAdt = self.beta * SA * phys_lam - (1 - self.alpha) * self.rho * IA - self.alpha * self.mu * IA + dOpi_Idt
        dIPdt = self.beta * self.VE_beta * SP * phys_lam - (1 - self.alpha * self.VE_death) * self.rho * IP - self.alpha * self.VE_death * self.mu * IP - dOpi_Idt

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

# %%
