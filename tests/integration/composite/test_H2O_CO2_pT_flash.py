"""Integration test for the general isobaric-isothermal flash.

The tests are conducting by performing calculations and comparing to values obtained
from ``thermo``.

"""
import csv
import pathlib
import sys
import unittest

import numpy as np

import porepy as pp


class Test_CO2_H2O_pT_flash(unittest.TestCase):

    l2_error_tolerance = 5e-3  # 0.5 %
    """Tolerance for L2-norm of residual between target and resulting values for each
    computed property."""
    abs_error_tolerance = 1e-2  # 1 %
    """Absolute error tolerance for local tests between single target and resulting
    values."""

    co2_fraction = 0.01
    h2o_fraction = 0.99

    def _sub_test_abs_error(self, result, target, property_name, msg=""):

        with self.subTest(
            f"Abs. Error subtest: {property_name}\nTolerance: {self.abs_error_tolerance}"
        ):
            residual = np.abs(result - target)
            self.assertTrue(
                np.all(residual <= self.abs_error_tolerance),
                f"Tolerance for <{property_name}> "
                + f"exceeded by {residual - self.abs_error_tolerance}."
                + f"\n{msg}",
            )

    def _sub_test_l2_error(self, result, target, property_name, msg=""):

        with self.subTest(
            f"L2-error subtest: <{property_name}>\nTolerance: {self.l2_error_tolerance}"
        ):
            res = result - target
            error = np.sqrt(np.dot(res, res))
            self.assertTrue(
                error <= self.l2_error_tolerance,
                f"Tolerance for <{property_name}> "
                + f"exceeded by {error - self.l2_error_tolerance}."
                + f"\n{msg}",
            )

    @unittest.skip("")
    def test_flash_only_gas(self):
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

        self._flash(
            row_ids=ids,
            p=p,
            T=T,
            y=y,
            Z_L=np.zeros(nc),
            Z_G=Z_G,
            x_h2o_L=np.zeros(nc),
            x_h2o_G=x_h2o_G,
            x_co2_L=np.zeros(nc),
            x_co2_G=x_co2_G,
            phi_h2o_L=np.zeros(nc),
            phi_h2o_G=phi_h2o_G,
            phi_co2_L=np.zeros(nc),
            phi_co2_G=phi_co2_G,
            test_gas=True,
            test_liquid=False,
            vectorize=True,
        )

    # @unittest.skip("Scenario with negative extended roots not covered.")
    def test_flash_only_liquid(self):
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

        self._flash(
            row_ids=ids,
            p=p,
            T=T,
            y=y,
            Z_L=Z_L,
            Z_G=np.zeros(nc),
            x_h2o_L=x_h2o_L,
            x_h2o_G=np.zeros(nc),
            x_co2_L=x_co2_L,
            x_co2_G=np.zeros(nc),
            phi_h2o_L=phi_h2o_L,
            phi_h2o_G=np.zeros(nc),
            phi_co2_L=phi_co2_L,
            phi_co2_G=np.zeros(nc),
            test_gas=False,
            test_liquid=True,
            vectorize=False,
        )

    @unittest.skip("")
    def test_flash_2_phase(self):
        """Test file pr_data_thermo_isothermal_GL_h2o_co2.csv."""
        path = pathlib.Path(__file__).parent.resolve()

        results_liq_file = open(str(path) + "/pr_data_thermo_isothermal_GL_h2o_co2.csv")
        results_liq = csv.reader(results_liq_file, delimiter=",")
        _ = next(results_liq)  # get rid of header

        ids = []
        p = []
        T = []
        y = []
        x_h2o_L = []
        x_h2o_G = []
        x_co2_L = []
        x_co2_G = []
        phi_h2o_L = []
        phi_h2o_G = []
        phi_co2_L = []
        phi_co2_G = []
        Z_L = []
        Z_G = []

        for datarow in results_liq:

            ids.append(int(datarow[0]))
            p.append(float(datarow[1]))
            T.append(float(datarow[2]))
            y.append(float(datarow[3]))
            x_h2o_L.append(float(datarow[4]))
            x_co2_L.append(float(datarow[5]))
            phi_h2o_L.append(float(datarow[6]))
            phi_co2_L.append(float(datarow[7]))
            Z_L.append(float(datarow[8]))
            x_h2o_G.append(float(datarow[9]))
            x_co2_G.append(float(datarow[10]))
            phi_h2o_G.append(float(datarow[11]))
            phi_co2_G.append(float(datarow[12]))
            Z_G.append(float(datarow[13]))

        ids = np.array(ids, dtype=int)
        p = np.array(p) * 1e-6  # scale to MPa
        T = np.array(T)
        y = np.array(y)
        x_h2o_L = np.array(x_h2o_L)
        x_co2_L = np.array(x_co2_L)
        phi_h2o_L = np.array(phi_h2o_L)
        phi_co2_L = np.array(phi_co2_L)
        Z_L = np.array(Z_L)
        x_h2o_G = np.array(x_h2o_G)
        x_co2_G = np.array(x_co2_G)
        phi_h2o_G = np.array(phi_h2o_G)
        phi_co2_G = np.array(phi_co2_G)
        Z_G = np.array(Z_G)

        assert (
            len(ids)
            == len(p)
            == len(T)
            == len(x_h2o_L)
            == len(x_co2_L)
            == len(phi_h2o_L)
            == len(phi_co2_L)
            == len(Z_L)
            == len(x_h2o_G)
            == len(x_co2_G)
            == len(phi_h2o_G)
            == len(phi_co2_G)
            == len(Z_G)
        ), "Data file: missing rows."
        results_liq_file.close()

        self._flash(
            row_ids=ids,
            p=p,
            T=T,
            y=y,
            Z_L=Z_L,
            Z_G=Z_G,
            x_h2o_L=x_h2o_L,
            x_h2o_G=x_h2o_G,
            x_co2_L=x_co2_L,
            x_co2_G=x_co2_G,
            phi_h2o_L=phi_h2o_L,
            phi_h2o_G=phi_h2o_G,
            phi_co2_L=phi_co2_L,
            phi_co2_G=phi_co2_G,
            test_gas=True,
            test_liquid=True,
            vectorize=False,
        )

    def _flash(
        self,
        row_ids,
        p,
        T,
        y,
        Z_L,
        Z_G,
        x_h2o_L,
        x_h2o_G,
        x_co2_L,
        x_co2_G,
        phi_h2o_L,
        phi_h2o_G,
        phi_co2_L,
        phi_co2_G,
        test_gas,
        test_liquid,
        vectorize,
    ):
        # Flag to vectorize the flash by stacking the values in the files row-wise.
        # This might cause some issues if single data rows cause trouble
        # for the whole flash, since the vectorized flash indicates convergence only if
        # every single row-wise flash converged.

        nc = len(p)
        if vectorize:
            nc_test = nc
        else:
            nc_test = 1

        y_result = np.zeros(nc)
        Z_L_result = np.zeros(nc)
        Z_G_result = np.zeros(nc)
        x_h2o_L_result = np.zeros(nc)
        x_h2o_G_result = np.zeros(nc)
        x_co2_L_result = np.zeros(nc)
        x_co2_G_result = np.zeros(nc)
        phi_h2o_L_result = np.zeros(nc)
        phi_h2o_G_result = np.zeros(nc)
        phi_co2_L_result = np.zeros(nc)
        phi_co2_G_result = np.zeros(nc)

        composition = pp.composite.PR_Composition(nc=nc_test)
        adsys = composition.ad_system

        phase_L = pp.composite.PR_Phase(adsys, False, name="L")
        phase_G = pp.composite.PR_Phase(adsys, True, name="G")

        h2o = pp.composite.H2O(adsys)
        co2 = pp.composite.CO2(adsys)

        composition.add_components([h2o, co2])
        composition.add_phases([phase_L, phase_G])

        # setting overall fractions
        adsys.set_variable_values(
            np.ones(nc_test) * self.h2o_fraction, [h2o.fraction_name], True, True, False
        )
        adsys.set_variable_values(
            np.ones(nc_test) * self.co2_fraction, [co2.fraction_name], True, True, False
        )

        # setting zero enthalpy, which does not matter for pT flash
        adsys.set_variable_values(
            np.zeros(nc_test), [composition.h_name], True, True, False
        )
        if vectorize:
            adsys.set_variable_values(p, [composition.p_name], True, True, False)
            adsys.set_variable_values(T, [composition.T_name], True, True, False)

        composition.initialize()

        flash = pp.composite.Flash(composition, auxiliary_npipm=False)
        flash.use_armijo = True
        flash.armijo_parameters["rho"] = 0.99
        flash.armijo_parameters["j_max"] = 50
        flash.armijo_parameters["return_max"] = True
        flash.newton_update_chop = 1.0
        flash.flash_tolerance = 1e-7
        flash.max_iter_flash = 50

        if vectorize:
            success = flash.flash("isothermal", "npipm", "rachford_rice", False, False)
            self.assertTrue(success, msg=f"Vectorized Flash did not succeed.")

            # compute values with obtained fractions
            composition.compute_roots()

            # extract values for l2 error check
            y_result = phase_G.fraction.evaluate(adsys).val
            Z_L_result = phase_L.eos.Z.val
            Z_G_result = phase_G.eos.Z.val
            x_h2o_L_result = phase_L.fraction_of_component(h2o).evaluate(adsys).val
            x_h2o_G_result = phase_G.fraction_of_component(h2o).evaluate(adsys).val
            x_co2_L_result = phase_L.fraction_of_component(co2).evaluate(adsys).val
            x_co2_G_result = phase_G.fraction_of_component(co2).evaluate(adsys).val
            phi_h2o_L_result = phase_L.eos.phi[h2o].val
            phi_h2o_G_result = phase_G.eos.phi[h2o].val
            phi_co2_L_result = phase_L.eos.phi[co2].val
            phi_co2_G_result = phase_G.eos.phi[co2].val

        else:
            for i in row_ids:
                with self.subTest(f"Flash test row ID: {i}."):
                    p_ = p[i]
                    T_ = T[i]
                    adsys.set_variable_values(
                        p_ * np.ones(nc_test), [composition.p_name], True, True, False
                    )
                    adsys.set_variable_values(
                        T_ * np.ones(nc_test), [composition.T_name], True, True, False
                    )

                    success = flash.flash(
                        "isothermal", "npipm", "rachford_rice", False, False
                    )
                    self.assertTrue(success, msg=f"Flash did not succeed. Row ID: {i}")

                    # compute values with obtained fractions
                    composition.compute_roots()

                    y_result[i] = adsys.get_variable_values(
                        [phase_G.fraction_name], True
                    )

                    Z_L_result[i] = phase_L.eos.Z.val
                    x_h2o_L_result[i] = (
                        phase_L.fraction_of_component(h2o).evaluate(adsys).val
                    )
                    x_co2_L_result[i] = (
                        phase_L.fraction_of_component(co2).evaluate(adsys).val
                    )
                    phi_h2o_L_result[i] = phase_L.eos.phi[h2o].val
                    phi_co2_L_result[i] = phase_L.eos.phi[co2].val

                    Z_G_result[i] = phase_G.eos.Z.val
                    x_h2o_G_result[i] = (
                        phase_G.fraction_of_component(h2o).evaluate(adsys).val
                    )
                    x_co2_G_result[i] = (
                        phase_G.fraction_of_component(co2).evaluate(adsys).val
                    )
                    phi_h2o_G_result[i] = phase_G.eos.phi[h2o].val
                    phi_co2_G_result[i] = phase_G.eos.phi[co2].val

        # global L2 error tests
        self._sub_test_l2_error(y_result, y, "Gas Fraction")

        if test_liquid:
            self._sub_test_l2_error(Z_L_result, Z_L, "Liquid Compressibility Factor")
            self._sub_test_l2_error(x_h2o_L_result, x_h2o_L, "Fraction H2O in Liquid")
            self._sub_test_l2_error(x_co2_L_result, x_co2_L, "Fraction CO2 in Liquid")
            self._sub_test_l2_error(
                phi_h2o_L_result, phi_h2o_L, "Fugacity Coefficient H2O in Liquid"
            )
            self._sub_test_l2_error(
                phi_co2_L_result, phi_co2_L, "Fugacity Coefficient CO2 in Liquid"
            )
        if test_gas:
            self._sub_test_l2_error(Z_G_result, Z_G, "Gas Compressibility Factor")
            self._sub_test_l2_error(x_h2o_G_result, x_h2o_G, "Fraction H2O in Gas")
            self._sub_test_l2_error(x_co2_G_result, x_co2_G, "Fraction CO2 in Gas")
            self._sub_test_l2_error(
                phi_h2o_G_result, phi_h2o_G, "Fugacity Coefficient H2O in Gas"
            )
            self._sub_test_l2_error(
                phi_co2_G_result, phi_co2_G, "Fugacity Coefficient CO2 in Gas"
            )

        # component-wise abs error tests
        for i in row_ids:
            y_i = y[i]
            y_i_r = y_result[i]
            p_i = p[i]
            T_i = T[i]
            self._sub_test_abs_error(
                y_i_r, y_i, "Gas Fraction", f"\nRow ID: {i}\np={p_i} ; T={T_i}"
            )

        for i in row_ids:
            p_i = p[i]
            T_i = T[i]
            subtest_message = f"\nRow ID: {i}\np={p_i} ; T={T_i}"

            if test_liquid:
                Z_i = Z_L[i]
                Z_i_r = Z_L_result[i]
                self._sub_test_abs_error(
                    Z_i_r, Z_i, "Liquid Compressibility Factor", subtest_message
                )

                x_i = x_h2o_L[i]
                x_i_r = x_h2o_L_result[i]
                self._sub_test_abs_error(
                    x_i_r, x_i, "Fraction H2O in Liquid", subtest_message
                )
                x_i = x_co2_L[i]
                x_i_r = x_co2_L_result[i]
                self._sub_test_abs_error(
                    x_i_r, x_i, "Fraction CO2 in Liquid", subtest_message
                )

                phi_i = phi_h2o_L[i]
                phi_i_r = phi_h2o_L_result[i]
                self._sub_test_abs_error(
                    phi_i_r,
                    phi_i,
                    "Fugacity coefficient H2O in Liquid",
                    subtest_message,
                )
                phi_i = phi_co2_L[i]
                phi_i_r = phi_co2_L_result[i]
                self._sub_test_abs_error(
                    phi_i_r,
                    phi_i,
                    "Fugacity coefficient CO2 in Liquid",
                    subtest_message,
                )

            if test_gas:
                Z_i = Z_G[i]
                Z_i_r = Z_G_result[i]
                self._sub_test_abs_error(
                    Z_i_r, Z_i, "Gas Compressibility Factor", subtest_message
                )

                x_i = x_h2o_G[i]
                x_i_r = x_h2o_G_result[i]
                self._sub_test_abs_error(
                    x_i_r, x_i, "Fraction H2O in Gas", subtest_message
                )
                x_i = x_co2_G[i]
                x_i_r = x_co2_G_result[i]
                self._sub_test_abs_error(
                    x_i_r, x_i, "Fraction CO2 in Gas", subtest_message
                )

                phi_i = phi_h2o_G[i]
                phi_i_r = phi_h2o_G_result[i]
                self._sub_test_abs_error(
                    phi_i_r, phi_i, "Fugacity coefficient H2O in Gas", subtest_message
                )
                phi_i = phi_co2_G[i]
                phi_i_r = phi_co2_G_result[i]
                self._sub_test_abs_error(
                    phi_i_r, phi_i, "Fugacity coefficient CO2 in Gas", subtest_message
                )


# if __name__ == "__main__":
#     unittest.main()


def main(out=sys.stderr, verbosity=2):
    loader = unittest.TestLoader()

    suite = loader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(out, verbosity=verbosity).run(suite)


if __name__ == "__main__":
    path = str(pathlib.Path(__file__).parent.resolve()) + "/test_results.txt"
    with open(path, "w") as f:
        main(f)
