import numpy as np
import numpy.typing as npt
import scipy.constants  # type: ignore
import warnings
from collections import UserDict

from collections import UserDict


class CoefficientsDict(UserDict):
    def __init__(self, values: npt.ArrayLike):
        values = np.asarray(values)
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
        T_range: tuple[float, ...],
        coeffs: list[CoefficientsDict],
    ):
        if len(T_range) != len(coeffs) + 1:
            raise ValueError(
                "Number of temperature ranges must be one more than the number of coefficients."
            )
        self.T_range = T_range
        self.coeffs = coeffs

    def __call__(self, T: float) -> CoefficientsDict:
        for i, T_bound in enumerate(self.T_range):
            if T <= T_bound:
                if i == 0:
                    warnings.warn(
                        "Temperature is below the lowest temperature of the Shomate coefficients."
                    )
                    return self.coeffs[i]
                return self.coeffs[i - 1]
        else:
            warnings.warn(
                "Temperature is above the highest temperature of the Shomate coefficients."
            )
            return self.coeffs[-1]


class constShomateCoefficients(ShomateCoefficients):
    def __init__(self, coeffs: list[CoefficientsDict]):
        self.T_range = (0, np.inf)
        self.coeffs = coeffs

    def __call__(self, T: float) -> CoefficientsDict:
        return self.coeffs[0]


class Shomate:
    def __init__(
        self,
        name_str: str,
        Shomate_coeffs: ShomateCoefficients,
        H_0_ref: float,
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
        coeffs = self.coeffs(T)
        c_P = (
            coeffs["A"]
            + coeffs["B"] * t
            + coeffs["C"] * t**2
            + coeffs["D"] * t**3
            + coeffs["E"] / t**2
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
        coeffs = self.coeffs(T)
        S = (
            coeffs["A"] * np.log(t)
            + coeffs["B"] * t
            + coeffs["C"] * t**2 / 2
            + coeffs["D"] * t**3 / 3
            - coeffs["E"] / (2 * t**2)
            + coeffs["G"]
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
        coeffs = self.coeffs(T)
        H = (
            coeffs["A"] * t
            + coeffs["B"] * t**2 / 2
            + coeffs["C"] * t**3 / 3
            + coeffs["D"] * t**4 / 4
            - coeffs["E"] / t
            + coeffs["F"]
            - coeffs["H"]
        )
        return H * 1000  # kJ/mol -> J/mol

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

    def Delta_f_H_0(
        self,
        nus: npt.ArrayLike,
        elements: list["ElementShomate"],
        T: float,
    ) -> float:
        """calculates the formation enthalpy of the species from the elements

        Parameters
        ----------
        T : float
            Temperature in Kelvin

        Returns
        -------
        float
            enthalpydifference to reference state in J/mol
        """
        nus = np.asarray(nus)
        return sum([nu * element.H_0(T) for nu, element in zip(nus, elements)])


class ElementShomate(Shomate):
    def __init__(
        self,
        name_str: str,
        Shomate_coeffs: ShomateCoefficients,
    ):
        super().__init__(name_str, Shomate_coeffs, 0)

    def Delta_f_H_0(self, T: float) -> float:  # type: ignore
        """calculates the formation enthalpy of the species

        Parameters
        ----------
        T : float
            Temperature in Kelvin

        Returns
        -------
        float
            enthalpydifference to reference state in J/mol
        """
        return 0


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
    nus = np.asarray(nus)
    return sum(
        [nu * component.Delta_f_H_0(T) for nu, component in zip(nus, components)]
    )


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
    nus = np.asarray(nus)
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
    T_range=(100, 500, 2000),
    coeffs=[
        CoefficientsDict(
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
        CoefficientsDict(
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
    ],
)

H2_c = ShomateCoefficients(
    T_range=(298, 1000, 2500),
    coeffs=[
        CoefficientsDict(
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
        CoefficientsDict(
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
    ],
)

NH3_c = ShomateCoefficients(
    T_range=(298, 1400, 6000),
    coeffs=[
        CoefficientsDict(
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
        CoefficientsDict(
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
    ],
)

N2 = ElementShomate("N2", N2_c)
H2 = ElementShomate("H2", H2_c)
NH3 = Shomate("NH3", NH3_c, H_0_ref=-45.90e3)


def main():
    print("N2")
    print(f"c_p(600)={NH3.c_P(600)}")
    print(f"S_0(600)={NH3.S_0(600)}")
    print(f"DH_0(600)={NH3.Delta_H_0(600)}")
    print("H2")
    print(f"c_p(300)={H2.c_P(300)}")
    print(f"S_0(300)={H2.S_0(300)}")
    print(f"DH_0(600)={H2.Delta_H_0(600)}")
    print("All these values match the NIST tables.")
    print("Test reaction enthalpy")
    print("N2->N2")
    print(Delta_f_H_0_N2 := Delta_R_H_0(298.15, [-1, 1], [N2, N2]))
    print("H2->H2")
    print(Delta_f_H_0_H2 := Delta_R_H_0(298.15, [-1, 1], [H2, H2]))
    print("1/2 N2 + 3/2 H2 -> NH3")
    print(Delta_f_H_0_NH3 := Delta_R_H_0(298.15, [-1 / 2, -3 / 2, 1], [N2, H2, NH3]))

    print(Delta_R_H_0(1000, [-1 / 2, -3 / 2, 1], [N2, H2, NH3]))


if __name__ == "__main__":
    main()
