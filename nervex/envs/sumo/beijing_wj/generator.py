import numpy as np
import math


class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated, mode):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps
        self.mode = mode
        assert self.mode in ['normal', 'congestion']
        self.route_prob = 0.2

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        if self.mode == 'normal':
            timings = np.random.uniform(0, 1, self._n_cars_generated)
        elif self.mode == 'congestion':
            timings = np.random.normal(0, 1, self._n_cars_generated)
            timings += np.abs(timings.min())
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        #dict_routes = get_routes()
        dict_routes = {
            'route1': "htxdj_e7 htxdj_e6 htxdj_e5 htxdj_e4 htxdj_e3 htxdj_e3.94 htxdj_e1 -72631490#2 -72631490#2.115 -72631490#2.115.131 -72631490#1 -72631490#0",
            'route2': "72631490#1 72631490#2 htxdj_w1 htxdj_w2 htxdj_w2.141 htxdj_w3 htxdj_w4 htxdj_w5 htxdj_w6 htxdj_w7",
            'route3': "wjj_s1 wjj_s2 wjj_s3 wjj_s4 wjj_s5 wjj_s6 wjj_s7 wjj_s8 wjj_s9 wjj_s10",
            'route4': "wjj_n9 wjj_n8 wjj_n7 wjj_n6 wjj_n5 wjj_n4 wjj_n3 534224179#4 wjj_n1",
            'route5': "haxl_e8 haxl_e7 haxl_e6 haxl_e5 haxl_e4 haxl_e3",
            'route6': "haxl_w3 haxl_w4 haxl_w5 haxl_w6 haxl_w7 haxl_w8 haxl_w9 haxl_w10",

            'route7': "haxl_e8 haxl_e7 wjj_n5 wjj_n4 wjj_n3 534224179#4 wjj_n1",
            'route8': "haxl_e8 haxl_e7 wjj_s6 wjj_s7 wjj_s8 wjj_s9 wjj_s10",
            'route9': "haxl_w3 haxl_w4 haxl_w5 haxl_w6 wjj_s6 wjj_s7 wjj_s8 wjj_s9 wjj_s10",
            'route10': "haxl_w3 haxl_w4 haxl_w5 haxl_w6 wjj_s6 wjj_s7 wjj_s8 wjj_s9 wjj_s10",

            'route11': "wjj_s1 wjj_s2 wjj_s3 htxdj_e4 htxdj_e3 htxdj_e3.94 htxdj_e1 -72631490#2 -72631490#2.115 -72631490#2.115.131 -72631490#1 -72631490#0",
            'route12': "wjj_s1 wjj_s2 wjj_s3 htxdj_w5 htxdj_w6 htxdj_w7",
            'route13': "htxdj_e7 htxdj_e6 htxdj_e5 wjj_s4 wjj_s5 haxl_e6 haxl_e5 haxl_e4 haxl_e3",

            'route14': "htxdj_e7 htxdj_e6 173019247#0 173019251 173019259#1 wjj_s5 wjj_s6 wjj_s7 wjj_s8 wjj_s9 wjj_s10",
            'route15': "wjj_s1 wjj_s2 173019256#0 173019407#0 -173019406#0 htxdj_w6 htxdj_w7",
            'route16': "haxl_w3 haxl_w4 haxl_w5 htxdj_w3 173019244#0 173019244#1 173019244#2 173019244#3 534224179#4 wjj_n1",
            'route17': "haxl_e8 haxl_e7 wjj_n5 173019255#0 173019255#1 173019250#1 htxdj_e3 htxdj_e3.94 haxl_e5 haxl_e4 haxl_e3",
        }
        from_edge_nodes = ['gneE189', 'gneE190', '-gneE62', 'hgzj_s1', 'lzxj_s1', 'lzxel_w1', 'gsbdj_w1', 'gzl_w1', 'lzdel_w1', 'lzdj_n3',
                           'qyl_w1', 'wjdl_e6', 'gneE187', 'gneE184', 'gneE180', 'gneE178', '-gneE62']
        to_edge_nodes = ['-gneE189', '-gneE190', 'gneE62', 'hgzj_n1', 'lzxj_n1', 'lzxel_e1', 'gsbdj_e1', 'gzl_e1', 'lzdel_e1', 'lzdj_s3',
                         'qyl_e1', 'wjdl_w6', '-gneE187', '-gneE184', '-gneE180', '-gneE178', 'gneE62']
        mid_nodes = ['lzxj_s6', 'lzzj_n4', 'hgbj_s5', 'htxj_n7', 'htxdj_e1', 'wjj_s5', 'hadl_e4', 'hadl_e6', 'gsndj_s4', 'hrj_s6', 'hadl_e2']
        from_nodes = from_edge_nodes + mid_nodes
        to_nodes = to_edge_nodes + mid_nodes
        from_nodes_len = len(from_nodes)
        to_nodes_len = len(to_nodes)
        route_names = list(dict_routes.keys())
        len_route_names = len(route_names)
        #<route id="W_N" edges="W2TL TL2N"/>
        str_routes = ['<route id="{}" edges="{}"/>\n'.format(k, v) for k, v in dict_routes.items()]
        str_routes = ''.join(str_routes)
        # produce the file for cars generation, one car per line
        with open("episode_routes_{}.rou.xml".format(self.mode), "w") as routes:
            print("""<routes>\n<vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />\n\n{}""".format(str_routes), file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                p = np.random.uniform()
                if p < self.route_prob:
                    route = route_names[np.random.choice(range(len_route_names))]
                    v_id = '{}_{}'.format(route, car_counter)
                    print('    <vehicle id="%s" type="standard_car" route="%s" depart="%s" departLane="random" departSpeed="10" />' % (v_id, route, step), file=routes)
                else:
                    car_id = 'trip_car_{}'.format(car_counter)
                    from_edge = from_nodes[np.random.choice(range(from_nodes_len))]
                    to_edge = to_nodes[np.random.choice(range(to_nodes_len))]
                    print('    <vehicles><trip id="{}" depart="{}" from="{}" to="{}" type="standard_car" departLane="random" departSpeed="10"/></vehicles>'.format(car_id, step, from_edge, to_edge), file=routes)

            print("</routes>", file=routes)


if __name__ == '__main__':
    handle = TrafficGenerator(200, 2000, mode='normal')  # normal, congestion
    #handle = TrafficGenerator(200, 1500, mode='congestion')
    handle.generate_routefile(1)
