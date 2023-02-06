"""Integration test for the general isobaric-isothermal flash.

The tests are conducting by performing calculations and comparing to values obtained
from ``thermo``.

"""
import csv
import pathlib
import unittest

import numpy as np

import porepy as pp


class Test_pT_flash(unittest.TestCase):

    flash_tolerance = 1e-7
    """Criterion for flash to converge."""
    error_tolerance = 2e-5
    """Absolute tolerance for L2-norm of residual between obtained values and thermo values."""

    def _compute_excess(self, residual):
        return self.error_tolerance - np.sqrt(
            residual * residual
        )  # component-wise L2 norm

    def _identify_excess(self, ids, residual):
        excess = self._compute_excess(residual)
        exceeding_at = excess < 0.0
        return ids[exceeding_at], np.abs(excess[exceeding_at])

    @unittest.skip("Skipped during development for performance.")
    def test_flash_only_gas(self):
        """Tests flash with data in file pr_data_thermo_isothermal_G_h2o_co2.csv ."""

        path = pathlib.Path(__file__).parent.resolve()

        results_gas_file = open(str(path) + "/pr_data_thermo_isothermal_G_h2o_co2.csv")
        results_gas = csv.reader(results_gas_file, delimiter=",")
        _ = next(results_gas)  # get rid of header

        ids = []
        p = []
        T = []
        x_h2o_G = []
        x_co2_G = []
        phi_h2o_G = []
        phi_co2_G = []
        Z_G = []

        for datarow in results_gas:

            ids.append(int(datarow[0]))
            p.append(float(datarow[1]))
            T.append(float(datarow[2]))
            x_h2o_G.append(float(datarow[4]))
            x_co2_G.append(float(datarow[5]))
            phi_h2o_G.append(float(datarow[6]))
            phi_co2_G.append(float(datarow[7]))
            Z_G.append(float(datarow[8]))

        ids = np.array(ids, dtype=int)
        p = np.array(p) * 1e-6  # scale to MPa
        T = np.array(T)
        x_h2o_G = np.array(x_h2o_G)
        x_co2_G = np.array(x_co2_G)
        phi_h2o_G = np.array(phi_h2o_G)
        phi_co2_G = np.array(phi_co2_G)
        Z_G = np.array(Z_G)

        assert (
            len(ids)
            == len(p)
            == len(T)
            == len(x_h2o_G)
            == len(x_co2_G)
            == len(phi_h2o_G)
            == len(phi_co2_G)
            == len(Z_G)
        ), "Data file: missing rows."
        results_gas_file.close()

        nc = len(ids)
        y = np.ones(nc)  # in this test everything should be gas

        composition = pp.composite.PR_Composition(nc=nc)
        adsys = composition.ad_system

        phase_L = pp.composite.PR_Phase(adsys, False, name="L")
        phase_G = pp.composite.PR_Phase(adsys, True, name="G")

        h2o = pp.composite.H2O(adsys)
        co2 = pp.composite.CO2(adsys)

        composition.add_components([h2o, co2])
        composition.add_phases([phase_L, phase_G])

        # setting overall fractions
        adsys.set_variable_values(
            np.ones(nc) * 0.99, [h2o.fraction_name], True, True, False
        )
        adsys.set_variable_values(
            np.ones(nc) * 0.01, [co2.fraction_name], True, True, False
        )

        # setting zero enthalpy, which does not matter for pT flash
        adsys.set_variable_values(np.zeros(nc), [composition.h_name], True, True, False)
        adsys.set_variable_values(p, [composition.p_name], True, True, False)
        adsys.set_variable_values(T, [composition.T_name], True, True, False)

        composition.initialize()

        # setting flasher
        flash = pp.composite.Flash(composition, auxiliary_npipm=False)
        flash.use_armijo = True
        flash.armijo_parameters["rho"] = 0.99
        flash.armijo_parameters["j_max"] = 150
        flash.newton_update_chop = 1.0
        flash.flash_tolerance = self.flash_tolerance

        success = flash.flash("isothermal", "npipm", "rachford_rice", False, False)

        self.assertTrue(success, msg=f"Flash did not succeed.")

        # compute values with obtained fractions
        composition.compute_roots()

        # checking only gas
        with self.subTest("Gas fractions sub test."):
            y_result = adsys.get_variable_values([phase_G.fraction_name], True)
            residual = y_result - y

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"Gas fractions exceed tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

        # checking compressibility
        with self.subTest("Compressibility factor sub test."):
            Z_result = phase_G.eos.Z.val
            residual = Z_G - Z_result

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"Gas compressibility factor exceeds tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

        # check phase composition
        with self.subTest("Water fraction in gas sub test."):
            x_h2o_G_result = phase_G.fraction_of_component(h2o).evaluate(adsys).val
            residual = x_h2o_G - x_h2o_G_result

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"Water fractions in gas exceed tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

        with self.subTest("CO2 fraction in gas sub test."):
            x_co2_G_result = phase_G.fraction_of_component(co2).evaluate(adsys).val
            residual = x_co2_G - x_co2_G_result

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"CO2 fractions in gas exceed tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

        # check fugacity coefficients
        with self.subTest("H2O fugacity coefficients in gas sub test."):
            phi_h20_result = phase_G.eos.phi[h2o].val
            residual = phi_h2o_G - phi_h20_result

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"H2O fugacity coefficients in gas exceed tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

        with self.subTest("CO2 fugacity coefficients in gas sub test."):
            phi_co2_result = phase_G.eos.phi[co2].val[0]
            residual = phi_co2_G - phi_co2_result

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"CO2 fugacity coefficients in gas exceed tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

    def test_flash_only_liq(self):
        """Flash results should yield only liquid for this test."""
        path = pathlib.Path(__file__).parent.resolve()

        results_liq_file = open(str(path) + "/pr_data_thermo_isothermal_L_h2o_co2.csv")
        results_liq = csv.reader(results_liq_file, delimiter=",")
        _ = next(results_liq)  # get rid of header

        ids = []
        p = []
        T = []
        x_h2o_L = []
        x_co2_L = []
        phi_h2o_L = []
        phi_co2_L = []
        Z_L = []

        for datarow in results_liq:

            ids.append(int(datarow[0]))
            p.append(float(datarow[1]))
            T.append(float(datarow[2]))
            x_h2o_L.append(float(datarow[4]))
            x_co2_L.append(float(datarow[5]))
            phi_h2o_L.append(float(datarow[6]))
            phi_co2_L.append(float(datarow[7]))
            Z_L.append(float(datarow[8]))

        ids = np.array(ids, dtype=int)
        p = np.array(p) * 1e-6  # scale to MPa
        T = np.array(T)
        x_h2o_L = np.array(x_h2o_L)
        x_co2_L = np.array(x_co2_L)
        phi_h2o_L = np.array(phi_h2o_L)
        phi_co2_L = np.array(phi_co2_L)
        Z_L = np.array(Z_L)

        assert (
            len(ids)
            == len(p)
            == len(T)
            == len(x_h2o_L)
            == len(x_co2_L)
            == len(phi_h2o_L)
            == len(phi_co2_L)
            == len(Z_L)
        ), "Data file: missing rows."
        results_liq_file.close()

        nc = len(ids)
        y = np.zeros(nc)  # in this test everything should be liquid

        composition = pp.composite.PR_Composition(nc=nc)
        adsys = composition.ad_system

        phase_L = pp.composite.PR_Phase(adsys, False, name="L")
        phase_G = pp.composite.PR_Phase(adsys, True, name="G")

        h2o = pp.composite.H2O(adsys)
        co2 = pp.composite.CO2(adsys)

        composition.add_components([h2o, co2])
        composition.add_phases([phase_L, phase_G])

        # setting overall fractions
        adsys.set_variable_values(
            np.ones(nc) * 0.99, [h2o.fraction_name], True, True, False
        )
        adsys.set_variable_values(
            np.ones(nc) * 0.01, [co2.fraction_name], True, True, False
        )

        # setting zero enthalpy, which does not matter for pT flash
        adsys.set_variable_values(np.zeros(nc), [composition.h_name], True, True, False)
        adsys.set_variable_values(p, [composition.p_name], True, True, False)
        adsys.set_variable_values(T, [composition.T_name], True, True, False)

        composition.initialize()

        # setting flasher
        flash = pp.composite.Flash(composition, auxiliary_npipm=False)
        flash.use_armijo = True
        flash.armijo_parameters["rho"] = 0.99
        flash.armijo_parameters["j_max"] = 150
        flash.newton_update_chop = 1.0
        flash.flash_tolerance = self.flash_tolerance

        success = flash.flash("isothermal", "npipm", "rachford_rice", False, False)

        self.assertTrue(success, msg=f"Flash did not succeed.")

        # compute values with obtained fractions
        composition.compute_roots()

        # checking only gas
        with self.subTest("Gas fractions sub test."):
            y_result = adsys.get_variable_values([phase_G.fraction_name], True)
            residual = y_result - y

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"Gas fractions exceed tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

        # checking compressibility
        with self.subTest("Compressibility factor sub test."):
            Z_result = phase_L.eos.Z.val
            residual = Z_L - Z_result

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"Liquid compressibility factor exceeds tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

        # check phase composition
        with self.subTest("Water fraction in liquid sub test."):
            x_h2o_L_result = phase_L.fraction_of_component(h2o).evaluate(adsys).val
            residual = x_h2o_L - x_h2o_L_result

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"Water fractions in liquid exceed tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

        with self.subTest("CO2 fraction in liquid sub test."):
            x_co2_L_result = phase_L.fraction_of_component(co2).evaluate(adsys).val
            residual = x_co2_L - x_co2_L_result

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"CO2 fractions in liquid exceed tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

        # check fugacity coefficients
        with self.subTest("H2O fugacity coefficients in liquid sub test."):
            phi_h20_result = phase_L.eos.phi[h2o].val
            residual = phi_h2o_L - phi_h20_result

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"H2O fugacity coefficients in liquid exceed tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

        with self.subTest("CO2 fugacity coefficients in liquid sub test."):
            phi_co2_result = phase_L.eos.phi[co2].val[0]
            residual = phi_co2_L - phi_co2_result

            excess, exceeding_by = self._identify_excess(ids, residual)
            self.assertTrue(
                not np.any(excess),
                f"CO2 fugacity coefficients in liquid exceed tolerance of {self.error_tolerance} for {len(excess)}/{len(ids)} data row IDs:\n{excess}\nby up to {exceeding_by.max()}.",
            )

    def test_flash_2_phase(self):
        """Flash results should yield a liquid-gas mixture for this test."""
        pass


if __name__ == "__main__":
    unittest.main()
