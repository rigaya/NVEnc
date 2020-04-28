#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  -----------------------------------------------------------------------------------------
#  QSVEnc by rigaya
#  -----------------------------------------------------------------------------------------
#  The MIT License
# 
#  Copyright (c) 2011-2016 rigaya
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
# 
#  ------------------------------------------------------------------------------------------
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import time, re, threading
import statistics

DEFAULT_INTERVAL = 500
DEFAULT_KEEP_LENGTH = 60

tColorList = (
    (255,88,88),   #桃
    (255,255,255), #白
    (88,245,88),   #薄緑
    (254,46,247),  #薄ピンク
    (0,252,255),   #水色
    (116,223,0),   #若草
    (4,180,134),   #青～緑
    (254,154,46),  #橙
    (0,64,255),    #青
    (243,247,12),  #レモン
    (250,0,112),   #赤～ピンク
    (200,254,46),  #黄緑
    (104,138,8),   #若草(暗)
    (180,49,4),    #茶
    (200,200,200), #灰
    (172,91,245),  #紫
    (64,64,245),   #群青
    (255,0,0)      #赤
)

class PerfData:
    """
    各データを管理する
    aData ... データ配列
    sName ... データ名
    sUnit ... データの単位
    nId   ... データのインデックス
    curve ... グラフに登録するデータ情報
    plot  ... そのデータが使用するグラフ軸データ
    bShow ... そのデータを表示するか
    """
    aData = None
    sName = ""
    sUnit = ""
    nId = -1
    curve = None
    plot = None
    bShow = True

    def __init__(self, sName, sUnit, bShow):
        """
        sName ... データ名
        sUnit ... データの単位
        bShow ... データを表示するかどうか
        """
        assert isinstance(sName, str)
        assert isinstance(sUnit, str)
        self.aData    = []
        self.sName    = sName
        self.sUnit    = sUnit
        self.bShow    = bShow

class PerfYAxis:
    """
    y軸の情報を管理する
    """
    ymin = 0
    ymax = 10
    firstData = 0 #この軸を使用する最初のデータのインデックス
    sUnit = ""
    def __init__(self, ymin, ymax, sUnit, firstData):
        """
        ymin      ... 初期のyの下限値
        ymax      ... 初期のyの上限値
        sUnit     ... 軸の単位
        firstData ... この軸を使用する最初のデータのインデックス
        """
        #単位が"%"なら 0% ～ 100%の範囲を常に表示すれば良い
        self.ymin = 0   if sUnit == "%" else ymin
        self.ymax = 100 if sUnit == "%" else ymax
        self.sUnit = sUnit
        self.firstData = firstData

class PerfMonitor:
    """
    グラフ全体を管理する
    aXdata         ... [float]    x軸のデータ
    aPerfData      ... [PerfData] y軸の様々なデータ
    dictYAxis      ... { str : PerfYAxis } 単位系ごとの軸データ
    win            ... グラフデータ
    basePlot       ... 親となる軸
    xmin, xmax     ... x軸の範囲
    xkeepLength    ... x軸のデータ保持範囲
    nInputInterval ... 読み込み間隔
    timer          ... 読み込み用タイマー
    nCheckRangeCount    ... Y軸の値域をチェックした回数
    nCheckRangeInterval ... Y軸の値域をチェック間隔
    """
    aXdata = []
    aPerfData = []
    dictYAxis = { }
    win = None
    basePlot = None
    xmax = 5
    xmin = 0
    xkeepLength = 30
    nInputInterval = 200
    timer = None
    nCheckRangeCount = 0
    nCheckRangeInterval = 4

    def __init__(self, nInputInterval, xkeepLength=30):
        self.nInputInterval = nInputInterval
        self.xkeepLength = xkeepLength

    def addData(self, prefData):
        assert isinstance(prefData, PerfData)
        prefData.nId = len(self.aPerfData)
        if prefData.bShow:
            #色を選択してPenを生成
            plotPen = pg.mkPen(color=tColorList[len(self.aPerfData)])
            if len(self.aPerfData) == 0:
                #データがひとつもないとき
                #まず軸情報を初期化
                perfDataYAxis = PerfYAxis(0, 10, prefData.sUnit, len(self.aPerfData))
                #ウィンドウを初期化
                self.win = pg.PlotWidget()
                pg.setConfigOptions(antialias = True)
                self.win.resize(540, 500)
                self.win.setWindowTitle('QSVEncC Performance Monitor')
                self.win.show()
                
                #親として作成
                prefData.plot = self.win.plotItem
                prefData.plot.showAxis('right')
                prefData.plot.setLabel('bottom', 'time', units='s')
                prefData.plot.setLabel('left', text=None, units=prefData.sUnit)
                prefData.plot.addLegend() #凡例を表示(親の軸に追加されたデータは凡例に自動的に追加される)
                prefData.plot.showGrid(x = True, y = True) #メモリ線をグラフ内に表示
                prefData.curve = prefData.plot.plot(x=self.aXdata, y=prefData.aData, name=prefData.sName) #データを追加
                prefData.plot.setXRange(0, 15, update=False)
                prefData.plot.enableAutoRange('x', 0.95) #x軸は自動追従
                prefData.plot.getAxis('left').enableAutoSIPrefix(False) #y軸のSI接頭辞の付加を禁止

                #単位が"%"なら値域を0～100に固定する
                if prefData.sUnit == '%':
                    prefData.plot.setYRange(0, 100, 0, False)
                    prefData.plot.getAxis('left').setTickSpacing(25,12.5) #目盛間隔を指定
            
                #親の軸として登録
                self.basePlot = prefData.plot
                self.dictYAxis[prefData.sUnit] = perfDataYAxis
            else:
                if prefData.sUnit in self.dictYAxis:
                    #すでに同じ単位の軸が存在すれば、そこに追加する
                    perfDataYAxis = self.dictYAxis[prefData.sUnit]
                    prefData.plot = self.aPerfData[perfDataYAxis.firstData].plot
                    #親の軸の場合にはplotにはpg.PlotItemが、それ以外の場合にはpg.ViewBoxが入っているので注意
                    #データの追加方法が異なる
                    if isinstance(self.aPerfData[perfDataYAxis.firstData].plot, pg.PlotItem):
                        prefData.curve = self.aPerfData[perfDataYAxis.firstData].plot.plot(x=self.aXdata, y=prefData.aData, name=prefData.sName)
                    else:
                        prefData.curve = pg.PlotCurveItem(x=self.aXdata, y=prefData.aData, name=prefData.sName)
                        prefData.plot.addItem(prefData.curve)
                else:
                    #異なる単位の軸なら、新たに追加する
                    perfDataYAxis = PerfYAxis(0, 10, prefData.sUnit, len(self.aPerfData))
                    prefData.plot = pg.ViewBox()
                    if len(self.dictYAxis) >= 2:
                        #3つめ以降の軸は新たに作成する必要がある
                        axItem = pg.AxisItem('right')
                        axItem.enableAutoSIPrefix(False) #軸のSI接頭辞の付加を禁止
                        self.basePlot.layout.addItem(axItem, 2, len(self.dictYAxis)+1)
                        self.basePlot.scene().addItem(prefData.plot)
                        axItem.linkToView(prefData.plot)
                        prefData.plot.setXLink(self.basePlot)
                        axItem.setLabel(units=prefData.sUnit)
                    else:
                        self.basePlot.showAxis('right')
                        self.basePlot.scene().addItem(prefData.plot)
                        self.basePlot.getAxis('right').enableAutoSIPrefix(False) #軸のSI接頭辞の付加を禁止
                        self.basePlot.getAxis('right').linkToView(prefData.plot)
                        prefData.plot.setXLink(self.basePlot)
                        self.basePlot.getAxis('right').setLabel(units=prefData.sUnit)
                    
                    self.dictYAxis[prefData.sUnit] = perfDataYAxis
                    prefData.curve = pg.PlotCurveItem(x=self.aXdata, y=prefData.aData, name=prefData.sName)
                    prefData.plot.addItem(prefData.curve)
                    prefData.plot.enableAutoRange('y', False)
                
                #デフォルトの軸以外は凡例に追加されていないので、ここに凡例に追加
                if prefData.plot != self.basePlot:
                    self.basePlot.legend.addItem(pg.PlotDataItem(pen=plotPen), prefData.sName)

            prefData.curve.setPen(plotPen)
        
        #データをaPerfDataに追加
        self.aPerfData.append(prefData)

        #データを読み込むタイマー
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.run)
        self.timer.start(max(self.nInputInterval-25, 10))

    def parse_input_line(self, line):
        elems = line.rstrip().split(",")
        try:
            current_time = float(elems[0])
            self.aXdata.append(current_time)
            for i in range(1, len(elems)):
                value = float(elems[i])
                self.aPerfData[i-1].aData.append(value)
        except:
            pass

    def run(self):
        line = sys.stdin.readline()
        self.parse_input_line(line)

        #x軸の範囲を取得
        xmin = min(self.aXdata)
        xmax = max(self.aXdata)

        #指定以上に範囲が長ければ削除
        removeData = xmax - xmin > self.xkeepLength + 3
        if removeData:
            self.aXdata = self.aXdata[1:]
            xmin = min(self.aXdata)

        for data in self.aPerfData:
            assert isinstance(data, PerfData)
            if removeData:
                data.aData = data.aData[1:]

            if data.bShow:
                #単位が"%"の場合は 0 - 100の固定でよい
                if self.nCheckRangeCount == 0 and data.sUnit != "%":
                    #自分の単位系全体について調整
                    perfDataYAxis = self.dictYAxis[data.sUnit]
                    ymin = min(data.aData)
                    ymax = max(data.aData)
                    ymedian = statistics.median(data.aData)
                    perfDataYAxis.ymin = min(perfDataYAxis.ymin, ymin)
                    if perfDataYAxis.ymax < ymax or ymax * 2.0 < perfDataYAxis.ymax:
                        perfDataYAxis.ymax = ymax * 1.05
                        #ある程度データがたまったら、medianにより上限を制限する
                        if len(self.aXdata) >= 60:
                            perfDataYAxis.ymax = min(perfDataYAxis.ymax, ymedian * 8)
                        data.plot.setYRange(perfDataYAxis.ymin, perfDataYAxis.ymax, 0, False)

                #更新されたデータを設定、自動的に反映される
                data.curve.setData(x=self.aXdata, y=data.aData)
                #
                if data.sUnit != self.aPerfData[0].sUnit:
                    data.plot.setGeometry(self.basePlot.vb.sceneBoundingRect())
                    data.plot.linkedViewChanged(self.basePlot.vb, data.plot.XAxis)
        
        #Y軸の値域チェックは4回に一回しか行わない
        self.nCheckRangeCount = (self.nCheckRangeCount + 1) % self.nCheckRangeInterval

if __name__ == "__main__":
    import sys
    pg.mkQApp()

    nInterval = DEFAULT_INTERVAL
    nKeepLength = DEFAULT_KEEP_LENGTH

    #コマンドライン引数を受け取る
    iargc = 1
    while iargc < len(sys.argv):
        if sys.argv[iargc] == "-i":
            iargc += 1
            try:
                nInterval = int(sys.argv[iargc])
            except:
                nInterval = DEFAULT_INTERVAL
        if sys.argv[iargc] == "-xrange":
            iargc += 1
            try:
                nKeepLength = int(sys.argv[iargc])
            except:
                nKeepLength = DEFAULT_KEEP_LENGTH
        iargc += 1

    monitor = PerfMonitor(nInterval, nKeepLength)

    #ヘッダー行を読み込み
    line = sys.stdin.readline()
    elems = line.rstrip().split(",")

    #"()"内を「単位」として抽出するための正規表現
    r = re.compile(r'(.*)\s\((.+)\)')
    for counter in elems[1:]:
        #"()"内を「単位」として抽出
        m = r.search(counter)
        unit = "" if m == None else m.group(2)
        #name = "" if m == None else m.group(1)
        #データとして追加 (単位なしや平均は表示しない)
        monitor.addData(PerfData(counter, unit, m != None))
    
    #凡例を作成
    counter_names = []
    for data in monitor.aPerfData:
        if data.bShow:
            counter_names.append(data.sName)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()