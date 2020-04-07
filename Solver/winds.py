import numpy as np


class Winds:
    def __init__(self, alfa_real_deg, tws_ref, V_yacht,is_flat_profile=False):
        """
        Parameters
        ----------
        alfa_real - [deg] angle between true wind and direction of boat movement (including leeway)
        tws -  Free stream of true wind having velocity [m/s] at height z = 10 [m]
        roughness - Over smooth, open water, expect a value around 0.0002 m,
                    while over flat, open grassland  ~ 0.03 m,
                    cropland ~ 0.1-0.25 m, and brush or forest ~ 0.5-1.0 m
        deck_height - NOT USED // Deck height above water level, height of sail foot may be defined from deck level: -> z[0] = 0.8[m] -> height of boom above the deck

        """
        self.alfa_real = np.deg2rad(alfa_real_deg)  # [rad] angle between true wind and direction of boat movement (including leeway)
        self.tws_ref = tws_ref
        self.roughness = 0.05
        self.deck_height = 0
        self.V_yacht = V_yacht
        self.is_flat_profile = is_flat_profile

    def get_true_wind_speed_at_h(self, height):
        # calc_tws_at_h = lambda h: self.tws_ref * (np.log((self.deck_height + h) / self.roughness) / np.log(10 / self.roughness))
        # wind_speed = np.array([calc_tws_at_h(h) for h in heights])

        if self.is_flat_profile:
            tws_at_h = np.array([self.tws_ref * np.cos(self.alfa_real), self.tws_ref * np.sin(self.alfa_real), 0])
        else:
            wind_speed = self.tws_ref * (np.log((self.deck_height + height) / self.roughness) / np.log(10 / self.roughness))
            tws_at_h = np.array([wind_speed * np.cos(self.alfa_real), wind_speed * np.sin(self.alfa_real), 0])

        return tws_at_h

    def get_app_alfa_infs_at_h(self, tws_at_h):
        # alfa_yacht - angle between apparent wind and true wind
        # alfa_app_infs	- angle between apparent wind and direction of boat movement (including leeway)
        # (model of an 'infinite sail' is assumed == without induced wind velocity) and direction of boat movement (including leeway)

        tws_l = np.linalg.norm(tws_at_h)  # true wind speed - length of the vector
        # tws_l = np.sqrt(tws_at_h[0]*tws_at_h[0]+tws_at_h[1]*tws_at_h[1])  # true wind speed

        alfa_yacht = np.arccos((self.V_yacht * np.cos(self.alfa_real) + tws_l) /
                               np.sqrt(
                                   self.V_yacht*self.V_yacht
                                   + 2*self.V_yacht*tws_l*np.cos(self.alfa_real)
                                   + tws_l*tws_l))  # result in [rad]

        alpha_app_infs = self.alfa_real - alfa_yacht  # result in [rad]

        return alpha_app_infs

    def get_app_infs_at_h(self, tws_at_h):
        aw_infs = np.array([tws_at_h[0] + self.V_yacht, tws_at_h[1], tws_at_h[2]])
        return aw_infs


