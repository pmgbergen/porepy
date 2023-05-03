"""Integration test for the general isobaric-isothermal flash.

The tests are conducting by performing calculations and comparing to values obtained
from ``thermo``.

"""
import csv
import pathlib
import sys
import unittest

import HtmlTestRunner
import numpy as np

import porepy as pp


class Test_CO2_H2O_pT_flash(unittest.TestCase):

    co2_fraction = 0.01
    h2o_fraction = 0.99

    fugacity_tolerance = 10.0
    """Due to numerical issues with the exponential function in the formula."""

    def _sub_test_abs_error(self, result, target, property_name, tol, msg="", idx=None):

        with self.subTest(f"Abs. Error subtest (tol={str(tol)}): {str(property_name)}"):
            residual = np.abs(result - target)
            exceeded = residual > tol
            if idx is not None:
                exceeded = np.logical_and(exceeded, idx)
                residual_ = residual[idx]
            else:
                residual_ = residual
            self.assertTrue(
                np.all(np.logical_not(exceeded)),
                f"Absolute tolerance for <{str(property_name)}> "
                + f"exceeded at row:\n{str(np.where(exceeded)[0])}"
                + f"\nby\n{str(residual[exceeded])}"
                + f"\nError:\nMax: {str(np.max(residual_))} ; Min: {str(np.min(residual_))} ; "
                + f"Mean: {str(np.mean(residual_))}"
                + f"\nExceeding Values:\n{str(result[exceeded])}"
                + f"\n{str(msg)}",
            )

    def _sub_test_l2_error(self, result, target, property_name, tol, msg="", idx=None):

        with self.subTest(f"L2-error subtest (tol={str(tol)}): {str(property_name)}"):
            res = result - target
            if idx is not None:
                res = res[idx]
            error = np.sqrt(np.dot(res, res))
            self.assertTrue(
                error <= tol,
                f"L2-Tolerance for <{str(property_name)}> "
                + f"exceeded by {str(error - tol)}."
                + f"\nError: {str(error)}"
                + f"\n{str(msg)}",
            )

    # @unittest.skip("")
    def test_gas_hard(self):
        """Data from pr_data_thermo_isothermal_G_hard.csv"""

        l2_tol = 1e-7
        abs_tol = 1e-7
        path = pathlib.Path(__file__).parent.resolve()

        results_gas_file = open(str(path) + "/pr_data_thermo_isothermal_G_hard.csv")
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
            vectorize=False,
            l2_tol=l2_tol,
            abs_tol=abs_tol,
        )

    # @unittest.skip("")
    def test_liquid_hard(self):
        """Data from pr_data_thermo_isothermal_L_hard.csv"""

        l2_tol = 1e-7
        abs_tol = 1e-7
        path = pathlib.Path(__file__).parent.resolve()

        results_liq_file = open(str(path) + "/pr_data_thermo_isothermal_L_hard.csv")
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
            l2_tol=l2_tol,
            abs_tol=abs_tol,
        )

    # @unittest.skip("")
    def test_2_phase_hard(self):
        """Data from pr_data_thermo_isothermal_GL_hard.csv."""

        l2_tol = 1e-5
        abs_tol = 1e-5
        path = pathlib.Path(__file__).parent.resolve()

        results_liq_file = open(str(path) + "/pr_data_thermo_isothermal_GL_hard.csv")
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
            l2_tol=l2_tol,
            abs_tol=abs_tol,
        )

    # @unittest.skip("")
    def test_gas_easy(self):
        """Data from pr_data_thermo_isothermal_G_easy.csv"""

        l2_tol = 1e-7
        abs_tol = 1e-7
        path = pathlib.Path(__file__).parent.resolve()

        results_gas_file = open(str(path) + "/pr_data_thermo_isothermal_G_easy.csv")
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
            vectorize=False,
            l2_tol=l2_tol,
            abs_tol=abs_tol,
        )

    # @unittest.skip("")
    def test_liquid_easy(self):
        """Data from pr_data_thermo_isothermal_L_easy.csv"""

        l2_tol = 1e-7
        abs_tol = 1e-7
        path = pathlib.Path(__file__).parent.resolve()

        results_liq_file = open(str(path) + "/pr_data_thermo_isothermal_L_easy.csv")
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
            l2_tol=l2_tol,
            abs_tol=abs_tol,
        )

    # @unittest.skip("")
    def test_2_phase_easy(self):
        """Data from pr_data_thermo_isothermal_GL_easy.csv."""

        l2_tol = 1e-5
        abs_tol = 1e-5
        path = pathlib.Path(__file__).parent.resolve()

        results_liq_file = open(str(path) + "/pr_data_thermo_isothermal_GL_easy.csv")
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
            l2_tol=l2_tol,
            abs_tol=abs_tol,
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
        l2_tol,
        abs_tol,
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

        success_result = np.zeros(nc, dtype=bool)

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

        MIX = pp.composite.NonReactiveMixture(nc=nc_test)
        ads = MIX.AD.system

        phase_L = pp.composite.PR_Phase(ads, False, name="L")
        phase_G = pp.composite.PR_Phase(ads, True, name="G")

        h2o = pp.composite.H2O(ads)
        co2 = pp.composite.CO2(ads)

        MIX.set([h2o, co2], [phase_L, phase_G])

        # setting overall fractions
        ads.set_variable_values(
            np.ones(nc_test) * self.h2o_fraction, [h2o.fraction.name], True, True, False
        )
        ads.set_variable_values(
            np.ones(nc_test) * self.co2_fraction, [co2.fraction.name], True, True, False
        )

        MIX.AD.set_up()

        # setting zero enthalpy, which does not matter for pT flash
        ads.set_variable_values(
            np.zeros(nc_test), [MIX.AD.h.name], True, True, False
        )

        flash = pp.composite.FlashNR(MIX, auxiliary_npipm=False)
        flash.use_armijo = True
        flash.armijo_parameters["rho"] = 0.99
        flash.armijo_parameters["j_max"] = 50
        flash.armijo_parameters["return_max"] = True
        flash.newton_update_chop = 1.0
        flash.tolerance = 1e-8
        flash.max_iter = 70

        flash_erro_msg = dict()

        if vectorize:
            ads.set_variable_values(p, [MIX.AD.p.name], True, True, False)
            ads.set_variable_values(T, [MIX.AD.T.name], True, True, False)

            with self.subTest("Vectorized Flash convergence sub test."):
                success = flash.flash(
                    "pT", "npipm", "rachford_rice", False, False
                )
                self.assertTrue(success, msg=f"Vectorized Flash did not succeed.")

            # compute values with obtained fractions
            flash.post_process_fractions(False)
            MIX.AD.compute_properties_from_state(apply_smoother=False)

            # extract values for l2 error check
            y_result = phase_G.fraction.evaluate(ads).val
            Z_L_result = phase_L.eos.Z.val
            Z_G_result = phase_G.eos.Z.val
            x_h2o_L_result = phase_L.fraction_of_component(h2o).evaluate(ads).val
            x_h2o_G_result = phase_G.fraction_of_component(h2o).evaluate(ads).val
            x_co2_L_result = phase_L.fraction_of_component(co2).evaluate(ads).val
            x_co2_G_result = phase_G.fraction_of_component(co2).evaluate(ads).val
            phi_h2o_L_result = phase_L.eos.phi[h2o].val
            phi_h2o_G_result = phase_G.eos.phi[h2o].val
            phi_co2_L_result = phase_L.eos.phi[co2].val
            phi_co2_G_result = phase_G.eos.phi[co2].val

        else:
            for i in row_ids:
                # with self.subTest(f"Flash test row ID: {str(i)}:"):
                p_ = np.array([p[i]])
                T_ = np.array([T[i]])
                ads.set_variable_values(p_, [MIX.AD.p.name], True, True, False)
                ads.set_variable_values(T_, [MIX.AD.T.name], True, True, False)

                try:
                    success = flash.flash(
                        "pT", "npipm", "rachford_rice", False, False
                    )
                except Exception as err:
                    flash_erro_msg.update({i: str(err)})
                    success_result[i] = False
                    continue
                else:
                    success_result[i] = success

                # compute values with obtained fractions
                flash.post_process_fractions(False)
                MIX.AD.compute_properties_from_state(apply_smoother=False)

                y_result[i] = phase_G.fraction.evaluate(ads).val

                Z_L_result[i] = phase_L.eos.Z.val
                x_h2o_L_result[i] = (
                    phase_L.fraction_of_component(h2o).evaluate(ads).val
                )
                x_co2_L_result[i] = (
                    phase_L.fraction_of_component(co2).evaluate(ads).val
                )
                phi_h2o_L_result[i] = phase_L.eos.phi[h2o].val
                phi_co2_L_result[i] = phase_L.eos.phi[co2].val

                Z_G_result[i] = phase_G.eos.Z.val
                x_h2o_G_result[i] = (
                    phase_G.fraction_of_component(h2o).evaluate(ads).val
                )
                x_co2_G_result[i] = (
                    phase_G.fraction_of_component(co2).evaluate(ads).val
                )
                phi_h2o_G_result[i] = phase_G.eos.phi[h2o].val
                phi_co2_G_result[i] = phase_G.eos.phi[co2].val

        # row-wise success test
        if not vectorize:
            failed = np.where(np.logical_not(success_result))[0]
            msg_ = [f"\nRow {i}:\n{errmsg}" for i, errmsg in flash_erro_msg.items()]
            msg = ""
            for m in msg_:
                msg += str(m)
            with self.subTest("Un-vectorized Flash convergence sub test."):
                self.assertTrue(
                    np.all(success_result),
                    f"Flash failed at rows:\n{str(failed)}" + str(msg),
                )

        # global L2 error tests
        self._sub_test_l2_error(y_result, y, "Gas Fraction", l2_tol, idx=success_result)

        if test_liquid:
            self._sub_test_l2_error(
                Z_L_result,
                Z_L,
                "Liquid Compressibility Factor",
                l2_tol,
                idx=success_result,
            )
            self._sub_test_l2_error(
                x_h2o_L_result,
                x_h2o_L,
                "Fraction H2O in Liquid",
                l2_tol,
                idx=success_result,
            )
            self._sub_test_l2_error(
                x_co2_L_result,
                x_co2_L,
                "Fraction CO2 in Liquid",
                l2_tol,
                idx=success_result,
            )
            self._sub_test_l2_error(
                phi_h2o_L_result,
                phi_h2o_L,
                "Fugacity Coefficient H2O in Liquid",
                self.fugacity_tolerance,
                idx=success_result,
            )
            self._sub_test_l2_error(
                phi_co2_L_result,
                phi_co2_L,
                "Fugacity Coefficient CO2 in Liquid",
                self.fugacity_tolerance,
                idx=success_result,
            )
        if test_gas:
            self._sub_test_l2_error(
                Z_G_result,
                Z_G,
                "Gas Compressibility Factor",
                l2_tol,
                idx=success_result,
            )
            self._sub_test_l2_error(
                x_h2o_G_result,
                x_h2o_G,
                "Fraction H2O in Gas",
                l2_tol,
                idx=success_result,
            )
            self._sub_test_l2_error(
                x_co2_G_result,
                x_co2_G,
                "Fraction CO2 in Gas",
                l2_tol,
                idx=success_result,
            )
            self._sub_test_l2_error(
                phi_h2o_G_result,
                phi_h2o_G,
                "Fugacity Coefficient H2O in Gas",
                self.fugacity_tolerance,
                idx=success_result,
            )
            self._sub_test_l2_error(
                phi_co2_G_result,
                phi_co2_G,
                "Fugacity Coefficient CO2 in Gas",
                self.fugacity_tolerance,
                idx=success_result,
            )

        # component-wise abs error tests
        self._sub_test_abs_error(
            y_result, y, "Gas Fraction", abs_tol, idx=success_result
        )

        if test_liquid:
            self._sub_test_abs_error(
                Z_L_result,
                Z_L,
                "Liquid Compressibility Factor",
                abs_tol,
                idx=success_result,
            )

            self._sub_test_abs_error(
                x_h2o_L_result,
                x_h2o_L,
                "Fraction H2O in Liquid",
                abs_tol,
                idx=success_result,
            )
            self._sub_test_abs_error(
                x_co2_L_result,
                x_co2_L,
                "Fraction CO2 in Liquid",
                abs_tol,
                idx=success_result,
            )

            self._sub_test_abs_error(
                phi_h2o_L_result,
                phi_h2o_L,
                "Fugacity coefficient H2O in Liquid",
                self.fugacity_tolerance,
                idx=success_result,
            )
            self._sub_test_abs_error(
                phi_co2_L_result,
                phi_co2_L,
                "Fugacity coefficient CO2 in Liquid",
                self.fugacity_tolerance,
                idx=success_result,
            )
        if test_gas:
            self._sub_test_abs_error(
                Z_G_result,
                Z_G,
                "Gas Compressibility Factor",
                abs_tol,
                idx=success_result,
            )

            self._sub_test_abs_error(
                x_h2o_G_result,
                x_h2o_G,
                "Fraction H2O in Gas",
                abs_tol,
                idx=success_result,
            )
            self._sub_test_abs_error(
                x_co2_G_result,
                x_co2_G,
                "Fraction CO2 in Gas",
                abs_tol,
                idx=success_result,
            )

            self._sub_test_abs_error(
                phi_h2o_G_result,
                phi_h2o_G,
                "Fugacity coefficient H2O in Gas",
                self.fugacity_tolerance,
                idx=success_result,
            )
            self._sub_test_abs_error(
                phi_co2_G_result,
                phi_co2_G,
                "Fugacity coefficient CO2 in Gas",
                self.fugacity_tolerance,
                idx=success_result,
            )


# if __name__ == "__main__":
#     unittest.main()


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # # TEXT Runner
    # path = str(pathlib.Path(__file__).parent.resolve()) + "/results.txt"
    # with open(path, "w") as f:
    #     unittest.TextTestRunner(f, verbosity=2).run(suite)

    # # HTML Runner
    path = str(pathlib.Path(__file__).parent.resolve()) + "/test_results/"
    HtmlTestRunner.HTMLTestRunner(output=path, verbosity=2).run(suite)
