import numpy as np
import numpy.typing as npt
import scipy.constants
import warnings
from collections import UserDict

from collections import UserDict


class CoefficientsDict(UserDict):
    def __init__(self, values: npt.ArrayLike):
        if len(values) != 8:
            raise ValueError("Input list must have exactly 8 values.")
        super().__init__()
        keys = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for key, value in zip(keys, values):
            self.data[key] = value

    def __setitem__(self, key: str, value: float):
        if key not in self.data:
            raise KeyError("Invalid key. Only keys 'A' through 'H' are allowed.")
        super().__setitem__(key, value)

    def numpy(self):
        return np.array(list(self.data.values()))


class ShomateCoefficients:
    def __init__(
        self,
        lowest_T: float,
        mid_T: float,
        highest_T: float,
        low_coeffs: CoefficientsDict,
        high_coeffs: CoefficientsDict,
    ):
        self.lowest_T = lowest_T
        self.mid_T = mid_T
        self.highest_T = highest_T
        self.low_coeffs = low_coeffs
        self.high_coeffs = high_coeffs

    def __call__(self, T: float) -> CoefficientsDict:
        if T < self.lowest_T:
            warnings.warn(
                "Temperature is below the lowest temperature of the Shomate coefficients.Returning low T parametrization"
            )
            return self.low_coeffs
        elif T < self.mid_T:
            return self.low_coeffs
        elif T < self.highest_T:
            return self.high_coeffs
        else:
            warnings.warn(
                "Temperature is above the highest temperature of the Shomate coefficients. Returning high T parametrization"
            )
            return self.high_coeffs


class constShomateCoefficients(ShomateCoefficients):
    def __init__(self, coeffs: CoefficientsDict):
        self.lowest_T = (0,)
        self.mid_T = (np.nan,)
        self.highest_T = (np.inf,)
        self.coeffs = coeffs

    def __call__(self, T: float) -> CoefficientsDict:
        return self.coeffs


class Shomate:
    def __init__(
        self,
        name_str: str,
        Shomate_coeffs: ShomateCoefficients,
        H_0_ref: float = 0,
    ):
        self.name_str = name_str
        self.coeffs = Shomate_coeffs
        self.H_0_ref = H_0_ref

    def c_P(self, T: float) -> float:
        """claculates the heat capacity at constant pressure of the species

        Parameters
        ----------
        T : float
            Temperature in Kelvin

        Returns
        -------
        float
            heat capacity at constant pressure in J/mol/K
        """
        t = T / 1000
        c_P = (
            self.coeffs(T)["A"]
            + self.coeffs(T)["B"] * t
            + self.coeffs(T)["C"] * t**2
            + self.coeffs(T)["D"] * t**3
            + self.coeffs(T)["E"] / t**2
        )
        return c_P

    def S_0(self, T: float) -> float:
        """calculates the entropy at standard conditions of the species

        Parameters
        ----------
        T : float
            Temperature in Kelvin

        Returns
        -------
        float
            entropy at standard conditions in J/mol/K
        """
        t = T / 1000
        S = (
            self.coeffs(T)["A"] * np.log(t)
            + self.coeffs(T)["B"] * t
            + self.coeffs(T)["C"] * t**2 / 2
            + self.coeffs(T)["D"] * t**3 / 3
            - self.coeffs(T)["E"] / (2 * t**2)
            + self.coeffs(T)["G"]
        )
        return S

    def Delta_H_0(self, T: float) -> float:
        """calculates the difference of enthalpy to the reference state of the species

        Parameters
        ----------
        T : float
            Temperature in Kelvin

        Returns
        -------
        float
            enthalpydifference to reference state in J/mol
        """
        t = T / 1000
        H = (
            self.coeffs(T)["A"] * t
            + self.coeffs(T)["B"] * t**2 / 2
            + self.coeffs(T)["C"] * t**3 / 3
            + self.coeffs(T)["D"] * t**4 / 4
            - self.coeffs(T)["E"] / t
            + self.coeffs(T)["F"]
            - self.coeffs(T)["H"]
        )
        return H * 1000

    def H_0(self, T: float) -> float:
        """calculates the enthalpy  of the species

        Parameters
        ----------
        T : float
            Temperature in Kelvin

        Returns
        -------
        float
            enthalpy in J/mol
        """
        return self.Delta_H_0(T) + self.H_0_ref


def Delta_R_H_0(
    T: float,
    nus: npt.ArrayLike,
    components: list[Shomate],
) -> float:
    """calculates the reaction enthalphy of the components with the stoichiometric coefficients

    Parameters
    ----------
    T : float
        Temperature in Kelvin
    nus : npt.ArrayLike
        stoichiometric coefficients
    components : list[Shomate]
        reacting components

    Returns
    -------
    float
        reaction enthalphy in J/mol
    """
    return sum([nu * component.H_0(T) for nu, component in zip(nus, components)])


def Delta_R_S_0(
    T: float,
    nus: npt.ArrayLike,
    components: list[Shomate],
) -> float:
    """calculates the reaction entropy of the components with the stoichiometric coefficients

    Parameters
    ----------
    T : float
        Temperature in Kelvin
    nus : npt.ArrayLike
        stoichiometric coefficients
    components : list[Shomate]
        reacting components

    Returns
    -------
    float
        reaction entropy in J/mol/K
    """
    return sum([nu * component.S_0(T) for nu, component in zip(nus, components)])


def Delta_R_G_0(
    T: float,
    nus: npt.ArrayLike,
    components: list[Shomate],
) -> float:
    """calculates the reaction Gibbs energy of the components with the stoichiometric coefficients

    Parameters
    ----------
    T : float
        Temperature in Kelvin
    nus : npt.ArrayLike
        stoichiometric coefficients
    components : list[Shomate]
        reacting components

    Returns
    -------
    float
        reaction Gibbs energy in J/mol
    """
    return Delta_R_H_0(T, nus, components) - T * Delta_R_S_0(T, nus, components)


def K_std(
    T: float,
    nus: npt.ArrayLike,
    components: list[Shomate],
) -> float:
    return np.exp(-Delta_R_G_0(T, nus, components) / (scipy.constants.R * T))


N2_c = ShomateCoefficients(
    100,
    500,
    2000,
    low_coeffs=CoefficientsDict(
        [
            28.98641,
            1.853978,
            -9.647459,
            16.63537,
            0.000117,
            -8.671914,
            226.4168,
            0,
        ]
    ),
    high_coeffs=CoefficientsDict(
        [
            19.50583,
            19.88705,
            -8.598535,
            1.369784,
            0.527601,
            -4.935202,
            212.3900,
            0,
        ]
    ),
)

H2_c = ShomateCoefficients(
    298,
    1000,
    2500,
    low_coeffs=CoefficientsDict(
        [
            33.066178,
            -11.363417,
            11.432816,
            -2.772874,
            -0.158558,
            -9.980797,
            172.707974,
            0,
        ]
    ),
    high_coeffs=CoefficientsDict(
        [
            18.563083,
            12.257357,
            -2.859786,
            0.268238,
            1.977990,
            -1.147438,
            156.288133,
            0,
        ]
    ),
)

NH3_c = ShomateCoefficients(
    298,
    1400,
    6000,
    low_coeffs=CoefficientsDict(
        [
            19.99563,
            49.77119,
            -15.37599,
            1.921168,
            0.189174,
            -53.30667,
            203.8591,
            -45.89806,
        ]
    ),
    high_coeffs=CoefficientsDict(
        [
            52.02427,
            18.48801,
            -3.765128,
            0.248541,
            -12.45799,
            -85.53895,
            223.8022,
            -45.89806,
        ]
    ),
)

N2 = Shomate("N2", N2_c)
H2 = Shomate("H2", H2_c)
NH3 = Shomate("NH3", NH3_c, H_0_ref=-45.90e3)


if __name__ == "__main__":
    print(NH3.c_P(300))
    print(NH3.S_0(300))
    print(NH3.Delta_H_0(600))
    print(H2.c_P(300))
    print(H2.S_0(300))
    print(H2.Delta_H_0(600))
    print(Delta_R_H_0(298.15, [-1, -3, 2], [N2, H2, NH3]))
    print(Delta_R_H_0(600, [-1, -3, 2], [N2, H2, NH3]))
