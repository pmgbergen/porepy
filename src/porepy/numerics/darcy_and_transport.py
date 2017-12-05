class DarcyAndTransport():
    """ 
    Wrapper for a stationary Darcy problem and a transport problem
    on the resulting fluxes.
    The flow and transport inputs should be members of the
    Darcy and Parabolic classes, respectively.
    """

    def __init__(self, flow, transport):
        self.flow = flow
        self.transport = transport

    def solve(self):
        """ 
        Solve both problems. 
        """
        p = self.flow.step()
        self.flow.pressure()
        self.flow.discharge()

        s = self.transport.solve()
        return p, s['transport']

    def save(self, export_every=1):
        """ 
        Save for visualization. 
        """
        self.flow.save()
        self.transport.save(save_every=export_every)
