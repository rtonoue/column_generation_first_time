from typing import Iterable
import mip
import random


class BinPackingProblemMIP(mip.Model):
    def __init__(
        self,
        bin_size: int,
        item_size: Iterable[int] = None,
        num_items: int = 10,
        max_item_size=None,
    ):
        """ビンパッキング問題のインスタンスを初期化

        Args:
            bin_size (int): ビンの容量
            item_size (Iterable[int], optional): アイテムの大きさ. Defaults to None.
            num_items (int, optional): アイテム数. Defaults to 10.
            max_item_size (_type_, optional): アイテムの大きさの最大値(アイテムを乱数で生成するときのみ使用). Defaults to None.
        """
        super().__init__()  # mipのモデルオブジェクトを初期化
        self.bin_size = bin_size
        self.max_item_size = max_item_size
        self.num_items = num_items
        if item_size is None:
            if self.max_item_size is None:
                self.max_item_size = self.bin_size
            self.generate_rand_items(num_items, max_item_size)
        else:
            self.item_size = item_size
            self.num_items = len(self.item_size)
        assert bin_size >= max_item_size

    def generate_rand_items(self, N: int, S: int):
        """乱数でインスタンスを生成

        Args:
            N (int): アイテムの数
            S (int): アイテムのサイズの最大値
        """
        self.item_size = [random.randint(1, S) for i in range(N)]

    def __set_variables(self):
        """モデルに変数をセットする"""
        self.x = {
            (i, j): self.add_var(
                name="x_" + str(i) + "_" + str(j), lb=0, ub=1, var_type=mip.BINARY
            )
            for i in range(self.num_items)
            for j in range(self.num_items)
        }
        self.y = {
            j: self.add_var(name="y_" + str(j), lb=0, ub=1, var_type=mip.BINARY)
            for j in range(self.num_items)
        }

    def __set_constraints(self):
        """モデルに制約条件をセットする"""
        for i in range(self.num_items):
            self.add_constr(mip.xsum(self.x[i, j] for j in range(self.num_items)) == 1)

        for j in range(self.num_items):
            self.add_constr(
                mip.xsum(
                    self.item_size[i] * self.x[i, j] for i in range(self.num_items)
                )
                <= self.bin_size * self.y[j]
            )

    def __set_objective(self):
        self += mip.xsum(self.y[j] for j in range(self.num_items))

    def solve(self, timeLimit=100):
        """mip.Modelオブジェクトに最適化問題をセットしてCBCソルバーで解く

        Args:
            timeLimit (int, optional): 計算時間の上限[sec]. Defaults to 100.

        Returns:
            _type_: _description_
        """
        self.__set_variables()
        self.__set_constraints()
        self.__set_objective()
        self.optimize(max_seconds=timeLimit)
        return self.status


if __name__ == "__main__":
    bpp = BinPackingProblemMIP(bin_size=10, num_items=20, max_item_size=10)
    print(bpp.solve())
