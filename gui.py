import inspect
from tkinter.messagebox import *
import threading
from PIL import Image, ImageTk  # 图像控件
import numpy as np
import cv2
import collections
import operator
import twophase.solver as sv

import tkinter as tk
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from pyopengltk import OpenGLFrame

frame = []  # 视频流保存变量
facenone = [None] * 9  # 截取的一帧视频中把它分为9块保存的变量
center = [None] * 9  # 聚类后的中心颜色
face = []
cont = 0  # 面数变量
clockcolor = [None] * 9  # 一个面的颜色
totalcolor = np.array([[None] * 9] * 6)  # 识别到的所有颜色
frontcolor = [None] * 3  # 魔方归位色块
st = [None] * 54  #
ok = [None] * 19  # 解的的结果
speed = 5  # 动画速度

'''从这些样本中进行颜色的聚类'''
samplecolor = np.array([[84, 50, 40], [170, 163, 121], [75, 61, 124], [171, 168, 125], [87, 147, 48], [84, 161, 157],
                        [76, 68, 101], [82, 81, 172], [101, 179, 154], [173, 171, 151], [189, 183, 164],
                        [195, 189, 181], [161, 164, 145], [158, 158, 143], [183, 182, 184], [152, 160, 127],
                        [164, 171, 136], [168, 172, 156], [96, 40, 38], [108, 48, 44], [109, 44, 35], [87, 41, 31],
                        [94, 41, 30], [97, 41, 31], [82, 47, 30], [89, 49, 30], [95, 52, 30], [107, 188, 168],
                        [130, 205, 189], [125, 217, 191], [69, 178, 153], [68, 189, 164], [64, 208, 166],
                        [67, 168, 148],
                        [66, 177, 159], [62, 186, 160], [55, 162, 97], [95, 188, 122], [95, 199, 132], [46, 159, 94],
                        [46, 170, 91],
                        [42, 180, 89], [46, 153, 84], [46, 160, 87], [42, 165, 84], [47, 38, 137], [82, 69, 163],
                        [131, 115, 195], [43, 38, 139],
                        [47, 41, 155], [67, 58, 177], [43, 40, 122], [44, 40, 134], [47, 43, 150], [57, 59, 220],
                        [76, 75, 241],
                        [132, 133, 250], [55, 59, 228], [66, 68, 242], [107, 115, 250], [49, 57, 209], [52, 60, 226],
                        [58, 62, 242]])  # 颜色集 # 白，蓝，黄，绿，红，橙
labels = ['蓝', '白', '红', '白', '绿', '黄', '红', '橙', '黄', '白', '白', '白', '白', '白', '白', '白', '白', '白', '蓝', '蓝', '蓝', '蓝',
          '蓝', '蓝', '蓝', '蓝', '蓝', '黄', '黄', '黄', '黄', '黄', '黄', '黄', '黄', '黄', '绿', '绿', '绿', '绿', '绿', '绿', '绿', '绿',
          '绿', '红', '红', '红', '红', '红', '红', '红', '红', '红', '橙', '橙', '橙', '橙', '橙', '橙', '橙', '橙', '橙']

''' 绘制正方体面的顶点'''
vertices = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)
edges = ((0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7), (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7))
surfaces = ((0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4), (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6))
colors = ((1, 0, 0), (1, 1, 1), (1, 0.5, 0), (1, 1, 0), (0.68, 1, 0.18), (0, 0, 0.54))

'''动画字典'''

rot_cube_map = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}
rot_slice_map = {
    'L': (0, 0, 1), 'ML': (0, 1, 1), 'L.': (0, 2, 1), 'D': (1, 0, 1), 'MD': (1, 1, 1),
    'D.': (1, 2, 1), 'B': (2, 0, 1), 'MB': (2, 1, 1), 'B.': (2, 2, 1),
    'R.': (0, 0, -1), 'MR': (0, 1, -1), 'R': (0, 2, -1), 'U.': (1, 0, -1), 'MU': (1, 1, -1),
    'U': (1, 2, -1), 'F.': (2, 0, -1), 'MF': (2, 1, -1), 'F': (2, 2, -1),
}

'''从得到的解法到动画的步骤进行转换的变量'''
animation_list_ani = []
animation_list_sol = []
animation_list = []
recover = []

'''把魔方复原到和检测的魔方一个方位'''


def changtotherightfront(froncolor, rightcolor, upcolor):
    #     #     #原始 橙黄绿
    #     #     #橙白蓝 橙绿白 橙黄绿 橙蓝黄
    #     #     #白红蓝 白绿红 白橙绿 白蓝橙
    #     #     #蓝红黄 蓝白红 蓝橙白 蓝黄橙
    #     #     #绿红白 绿黄红 绿橙黄 绿白橙
    #     #     #黄绿橙 黄红绿 黄橙红 黄橙蓝
    #     #     #红黄蓝 红绿黄 红白绿 红蓝白
    tex = froncolor + rightcolor + upcolor
    colorfront = {
        'ogw': ['F', 'MF', 'F.'],
        'owb': ['F', 'MF', 'F.', 'F', 'MF', 'F.'],
        'oby': ['F', 'MF', 'F.', 'F', 'MF', 'F.', 'F', 'MF', 'F.'],
        'oyg': ['F', 'MF', 'F.', 'B', 'MB', 'B.'],

        'wog': ['D', 'MD', 'D.'],
        'wgr': ['D', 'MD', 'D.', 'F', 'MF', 'F.'],
        'wrb': ['D', 'MD', 'D.', 'F', 'MF', 'F.', 'F', 'MF', 'F.'],
        'wbo': ['D', 'MD', 'D.', 'B', 'MB', 'B.'],

        'byo': ['R', 'MR', 'R.'],
        'bow': ['R', 'MR', 'R.', 'F', 'MF', 'F.'],
        'bwr': ['R', 'MR', 'R.', 'F', 'MF', 'F.', 'F', 'MF', 'F.'],
        'bry': ['R', 'MR', 'R.', 'B', 'MB', 'B.'],

        'gyr': ['L', 'ML', 'L.'],
        'grw': ['L', 'ML', 'L.', 'F', 'MF', 'F.'],
        'gwo': ['L', 'ML', 'L.', 'F', 'MF', 'F.', 'F', 'MF', 'F.'],
        'goy': ['L', 'ML', 'L.', 'B', 'MB', 'B.'],

        'yrg': ['U', 'MU', 'U.'],
        'ygo': ['U', 'MU', 'U.', 'F', 'MF', 'F.'],
        'yob': ['U', 'MU', 'U.', 'F', 'MF', 'F.', 'F', 'MF', 'F.'],
        'ybr': ['U', 'MU', 'U.', 'B', 'MB', 'B.'],

        'rwg': ['U', 'MU', 'U.', 'U', 'MU', 'U.'],
        'rgy': ['U', 'MU', 'U.', 'U', 'MU', 'U.', 'F', 'MF', 'F.'],
        'ryb': ['U', 'MU', 'U.', 'U', 'MU', 'U.', 'F', 'MF', 'F.', 'F', 'MF', 'F.'],
        'rbw': ['U', 'MU', 'U.', 'U', 'MU', 'U.', 'B', 'MB', 'B.']}
    return colorfront[tex]


'''将得到的解法公式转换为魔方动画'''


def reslutcahnge(li):
    reslt = []
    print(len(li))
    for i in range(len(li)):
        tex = li[i:i + 2]
        if tex == 'L1':
            reslt += 'L'
        elif tex == 'L2':
            reslt += 'L' + 'L'
        elif tex == 'L3':
            reslt += 'L' + 'L' + 'L'
        elif tex == 'R1':
            reslt += 'R'
        elif tex == 'R2':
            reslt += 'R' + 'R'
        elif tex == 'R3':
            reslt += 'R' + 'R' + 'R'
        elif tex == 'U1':
            reslt += 'U'
        elif tex == 'U2':
            reslt += 'U' + 'U'
        elif tex == 'U3':
            reslt += 'U' + 'U' + 'U'
        elif tex == 'F1':
            reslt += 'F'
        elif tex == 'F2':
            reslt += 'F' + 'F'
        elif tex == 'F3':
            reslt += 'F' + 'F' + 'F'
        elif tex == 'B1':
            reslt += 'B'
        elif tex == 'B2':
            reslt += 'B' + 'B'
        elif tex == 'B3':
            reslt += 'B' + 'B' + 'B'
        elif tex == 'D1':
            reslt += 'D'
        elif tex == 'D2':
            reslt += 'D' + 'D'
        elif tex == 'D3':
            reslt += 'D' + 'D' + 'D'
    return reslt


'''将得到的解法公式转换为魔方动画'''


def showrubikecolor(li):
    reslt = []
    i = len(li)
    while i > 0:
        tex = li[i - 2:i]
        # reslt += tex
        i -= 2
        if tex == 'L1':
            reslt.append("R.")
        elif tex == 'L2':
            reslt.append('R.')
            reslt.append('R.')
        elif tex == 'L3':
            reslt.append('R.')
            reslt.append('R.')
            reslt.append('R.')
        elif tex == 'R1':
            reslt.append('L.')
        elif tex == 'R2':
            reslt.append('L.')
            reslt.append('L.')
        elif tex == 'R3':
            reslt.append('L.')
            reslt.append('L.')
            reslt.append('L.')
        elif tex == 'U1':
            reslt.append('D.')
        elif tex == 'U2':
            reslt.append('D.')
            reslt.append('D.')
        elif tex == 'U3':
            reslt.append('D.')
            reslt.append('D.')
            reslt.append('D.')
        elif tex == 'F1':
            reslt.append('B.')
        elif tex == 'F2':
            reslt.append('B.')
            reslt.append('B.')
        elif tex == 'F3':
            reslt.append('B.')
            reslt.append('B.')
            reslt.append('B.')
        elif tex == 'B1':
            reslt.append("F.")
        elif tex == 'B2':
            reslt.append("F.")
            reslt.append("F.")
        elif tex == 'B3':
            reslt.append("F.")
            reslt.append("F.")
            reslt.append("F.")
        elif tex == 'D1':
            reslt.append('U.')
        elif tex == 'D2':
            reslt.append('U.')
            reslt.append('U.')
        elif tex == 'D3':
            reslt.append('U.')
            reslt.append('U.')
            reslt.append('U.')
    return reslt


'''点击清空时将魔方复原到初始状态的动画'''


def rechange(li):
    i = len(li)
    print(i)
    reslut = []
    reli = {'L': 'R.', 'ML': 'MR', 'L.': 'R', 'D': 'U.', 'MD': 'MU',
            'D.': 'U', 'B': 'F.', 'MB': 'MF', 'B.': 'F',
            'R.': 'L', 'MR': 'ML', 'R': 'L.', 'U.': 'D', 'MU': 'MD',
            'U': 'D.', 'F.': 'B', 'MF': 'MB', 'F': 'B.'}
    while i > 0:
        i = i - 1
        tex = li[i]
        reslut.append(reli[tex])

    return reslut


'''魔方绘制类'''


class Cube():
    def __init__(self, id, N, scale):
        self.N = 3
        self.scale = scale
        self.init_i = [*id]
        self.current_i = [*id]  # 表示填充，一个变量值代替多个
        self.rot = [[1 if i == j else 0 for i in range(3)] for j in range(3)]

    def isAffected(self, axis, slice, dir):
        return self.current_i[axis] == slice

    def update(self, axis, slice, dir):

        if not self.isAffected(axis, slice, dir):
            return

        i, j = (axis + 1) % 3, (axis + 2) % 3
        for k in range(3):
            self.rot[k][i], self.rot[k][j] = -self.rot[k][j] * dir, self.rot[k][i] * dir

        self.current_i[i], self.current_i[j] = (
            self.current_i[j] if dir < 0 else self.N - 1 - self.current_i[j],
            self.current_i[i] if dir > 0 else self.N - 1 - self.current_i[i])

    def transformMat(self):
        scaleA = [[s * self.scale for s in a] for a in self.rot]
        scaleT = [(p - (self.N - 1) / 2) * 2.1 * self.scale for p in self.current_i]
        return [*scaleA[0], 0, *scaleA[1], 0, *scaleA[2], 0, *scaleT, 1]

    def draw(self, col, surf, vert, animate, angle, axis, slice, dir):

        glPushMatrix()
        if animate and self.isAffected(axis, slice, dir):
            glRotatef(angle * dir, *[1 if i == axis else 0 for i in range(3)])  # 围着这个坐标点旋转
        glMultMatrixf(self.transformMat())

        glBegin(GL_QUADS)
        for i in range(len(surf)):
            glColor3fv(colors[i])
            for j in surf[i]:
                glVertex3fv(vertices[j])
        glEnd()

        glPopMatrix()


'''pytkinkergl 这个框架下绘制魔方'''


class GLFrame(OpenGLFrame):
    global speed

    def initgl(self):
        self.animate = True
        self.rota = 0
        self.count = 0

        self.ang_x, self.ang_y, self.rot_cube = 0, 0, (0, 0)
        self.animate1Cube, self.animate_ang, self.animate_speed = False, 0, speed

        self.action = (0, 0, 0)
        glClearColor(1.0, 0.89, 0.77, 1.0)  # 背景黑色
        # glViewport(400, 400, 200, 200)  # 指定了视口的左下角位置

        glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
        glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()  # 恢复原始坐标
        gluPerspective(30, self.width / self.height, 0.1, 50.0)

        self.N = 3
        cr = range(self.N)
        self.cubes = [Cube((x, y, z), self.N, 1.5) for x in cr for y in cr for z in cr]

    def redraw(self):
        self.ang_x += self.rot_cube[0] * 2
        self.ang_y += self.rot_cube[1] * 2
        self.animate_speed = speed
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -40)
        glRotatef(self.ang_y, 0, 1, 0)
        glRotatef(self.ang_x, 1, 0, 0)
        gluLookAt(2, 2, 2,
                  0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.animate1Cube:
            if self.animate_ang >= 90:
                for cube in self.cubes:
                    cube.update(*self.action)
                self.animate1Cube, self.animate_ang = False, 0
        if not self.animate1Cube and animation_list:
            self.animate1Cube, self.action = True, rot_slice_map[animation_list[0]]
            del animation_list[0]

        for cube in self.cubes:
            cube.draw(colors, surfaces, vertices, self.animate, self.animate_ang, *self.action)
        if self.animate1Cube:
            self.animate_ang += self.animate_speed


'''一整个GUI'''


class TestGui(object):
    '''tkinterGUI初始化'''

    def __init__(self, init_window_name):
        super().__init__()
        self.init_window_name = init_window_name
        self.init_window_name.title("魔方复原")  # 设置窗口标题
        self.init_window_name.geometry('900x600')  # 设置窗口大小
        self.init_window_name.resizable(0, 0)

        screenWidth = self.init_window_name.winfo_screenwidth()  # 获取显示区域的宽度
        screenHeight = self.init_window_name.winfo_screenheight()  # 获取显示区域的高度
        width = 900  # 设定窗口宽度
        height = 600  # 设定窗口高度
        left = (screenWidth - width) / 2
        top = (screenHeight - height) / 2
        # 宽度x高度+x偏移+y偏移
        # 在设定宽度和高度的基础上指定窗口相对于屏幕左上角的偏移位置
        self.init_window_name.geometry("%dx%d+%d+%d" % (width, height, left, top))

        """ 图片显示 """
        # self.picture = tk.PhotoImage(file='img/line.gif')
        """ 点击右上角关闭窗体弹窗事件 """
        self.init_window_name.protocol('WM_DELETE_WINDOW', lambda: self.thread_it(self.clos_window))
        """ 画布创建 """
        # 画布控件
        self.canvasvedio = tk.Canvas(self.init_window_name, bg='black', width=300, height=300)  # 绘制画布
        self.canvasresult = tk.Canvas(self.init_window_name, bg='black', width=300, height=300)  # 绘制画布
        self.canvasclock = tk.Canvas(self.init_window_name, width=410, height=300)  # 绘制画布
        self.canvasfuyuan = tk.Canvas(self.init_window_name, bg='#FFE4C4', width=300, height=294)  # 绘制画布
        self.canvaslog = tk.Canvas(self.init_window_name, width=190, height=300)  # 绘制画布

        self.canvasvedio.place(x=0, y=0)
        self.canvasfuyuan.place(x=600, y=2)
        self.canvasresult.place(x=300, y=0)
        self.canvasclock.place(x=300, y=300)
        self.canvaslog.place(x=710, y=300)

        """ 操作按钮 """
        self.ok = tk.Button(self.init_window_name, text='识别', height=1, width=16, font=('行楷', 20, 'bold'), fg="white",
                            bg="#1E90FF", command=lambda: self.thread_it(self.colorregnize))
        self.ok.place(x=1, y=300)

        self.okk = tk.Button(self.init_window_name, text='确定', height=1, width=16, font=('行楷', 20, 'bold'), fg="white",
                             bg="#1E90FF", command=lambda: self.thread_it(self.getrgb))
        self.okk.place(x=1, y=375)

        self.clear = tk.Button(self.init_window_name, text='清空', height=1, width=16, font=('行楷', 20, 'bold'),
                               fg="white",
                               bg="#1E90FF", command=lambda: self.thread_it(self.clearcanvas))
        self.clear.place(x=1, y=450)

        self.adjust = tk.Button(self.init_window_name, text="复原", height=1, width=7, font=('行楷', 20, 'bold'),
                                fg="white",
                                bg="#1E90FF", command=lambda: self.thread_it(self.solver))
        self.adjust.place(x=1, y=525)

        self.animotion = tk.Button(self.init_window_name, text="动画", height=1, width=7, font=('行楷', 20, 'bold'),
                                   fg="white",
                                   bg="#1E90FF", command=self.fuyuan)
        self.animotion.place(x=163, y=525)

        # self.canvasclock.create_text(58, 90, text='此面正对你', fill='#2F4F4F', font=('微软雅黑', 10, 'bold'))

        """视频读取  使用的多线程  """
        self.thread_it(self.showvideo)
        self.thread_it(self.showfuyuan)
        '''显示魔方界面的框架初始化'''

    def showfuyuan(self):
        super().__init__()
        self.glframe = GLFrame(self.canvasfuyuan, width=300, height=300)
        self.glframe.place(x=0, y=0)
        self.glframe.animate = True

    '''多线程'''

    def thread_it(self, func, *args):
        """ 将函数打包进线程 """
        self.myThread = threading.Thread(target=func, args=args)
        self.myThread.setDaemon(True)  # 主线程退出就直接让子线程跟随退出,不论是否运行完成。
        self.myThread.start()

    '''结束线程'''

    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self, thread):
        self._async_raise(thread.ident, SystemExit)

    '''tkinter不能直接显示视频，需要把opencv得到的图片进行格式转换后显示'''

    def pic_change(self, pic):  # 转换图片格式
        cvimage = cv2.cvtColor(pic, cv2.COLOR_BGR2RGBA)  # 转换一下，不然gui不能显示出来
        pilImage = Image.fromarray(cvimage)
        tkImage = ImageTk.PhotoImage(image=pilImage)
        return tkImage

    '''好像没用到'''

    def drawline(self):
        self.canvasclock.create_rectangle(98, 5, 195, 410, outline='red', width=5)
        for i in range(3):
            self.canvasclock.create_text(210, i * 100 + 50, text=i + 1, fill='#7CCD7C', font=('微软雅黑', 15, 'bold'))

    '''点击识别按钮进行颜色的识别，这里先将图片分成9块，然后再进行聚类，K近邻等机器学习算法，然后得到每一块的颜色，这一步是识别，不一定准确，不准确可以多次识别'''

    def colorregnize(self):
        global clockcolor, frame
        colorbar = [None] * 9
        self.clock1 = frame[0:100, 0:  100]  # 大小为100*100
        self.clock2 = frame[0:100, 100:200]
        self.clock3 = frame[0:100, 200:300]

        self.clock4 = frame[100:200, 0:  100]
        self.clock5 = frame[100:200, 100:200]
        self.clock6 = frame[100:200, 200:300]

        self.clock7 = frame[200:300, 0:100]
        self.clock8 = frame[200:300, 100:200]
        self.clock9 = frame[200:300, 200:300]
        '''图片块保存'''
        face = [self.clock1, self.clock2, self.clock3, self.clock4, self.clock5, self.clock6, self.clock7,
                self.clock8, self.clock9]
        '''如果保存所有识别色块的变量未满（也就是还没识别完54块色块）'''
        if totalcolor[5][8] == None:
            for i in range(len(face)):
                image, counter, cen = self.color_quantization1(face[i], 1)  # 对颜色进行聚类
                facenone[i] = image
                center[i] = cen
            for i in range(len(center)):
                test_class = self.classify0(center[i], samplecolor, labels, 2)  # 对颜色进行近邻
                clockcolor[i] = test_class
                # print(test_class)
            for i in range(len(clockcolor)):  # 将识别得到的色块值进行RGB颜色映射，便于后续显示出来
                if clockcolor[i] == '红':
                    colorbar[i] = '#ff0000'
                elif clockcolor[i] == '橙':
                    colorbar[i] = '#FFA500'
                elif clockcolor[i] == '黄':
                    colorbar[i] = '#FFFF00'
                elif clockcolor[i] == '绿':
                    colorbar[i] = '#9ACD32'
                elif clockcolor[i] == '蓝':
                    colorbar[i] = '#191970'
                elif clockcolor[i] == '白':
                    colorbar[i] = '#F0FFF0'
            self.colorshow(colorbar)
            # print(cont)
        else:
            showwarning(title='提示', message='已经识别完成，请开始复原！')  # 如果识别完了就抛出警告
        # print(totalcolor)

    '''点击确定按钮时将进行颜色识别，这时说明你已经确认好了颜色'''

    def getrgb(self):
        colorbar = [None] * 9
        global totalcolor
        global cont
        if totalcolor[5][8] == None:
            for j in range(len(clockcolor)):
                totalcolor[cont][j] = clockcolor[j]
            totalcolor[cont][0], totalcolor[cont][2] = totalcolor[cont][2], totalcolor[cont][0]
            totalcolor[cont][5], totalcolor[cont][3] = totalcolor[cont][3], totalcolor[cont][5]
            totalcolor[cont][8], totalcolor[cont][6] = totalcolor[cont][6], totalcolor[cont][8]
            # print(totalcolor, cont)
            # print('这一面的颜色是：', clockcolor, cont)
            for i in range(9):
                if totalcolor[cont][i] == '红':
                    colorbar[i] = '#ff0000'
                elif totalcolor[cont][i] == '橙':
                    colorbar[i] = '#FFA500'
                elif totalcolor[cont][i] == '黄':
                    colorbar[i] = '#FFFF00'
                elif totalcolor[cont][i] == '绿':
                    colorbar[i] = '#9ACD32'
                elif totalcolor[cont][i] == '蓝':
                    colorbar[i] = '#191970'
                elif totalcolor[cont][i] == '白':
                    colorbar[i] = '#F0FFF0'
            # print(colorbar)
            # 将每一面展示出来
            if cont <= 2:
                self.showresult(self.canvasclock, cont * 100 + 10, 102, colorbar)
            if cont == 3:
                self.showresult(self.canvasclock, cont * 100 + 10, 102, colorbar)
            elif cont == 4:
                self.showresult(self.canvasclock, 200 + 10, 2, colorbar)
            elif cont == 5:
                self.showresult(self.canvasclock, 200 + 10, 202, colorbar)

            cont = cont + 1
            if cont == 6:
                cont = 0

        else:
            showwarning(title='提示', message='已经识别完成，请开始复原！')
            # print(totalcolor)

    '''清空画布，重回初始状态，开始下一次的识别'''

    def clearcanvas(self):
        global totalcolor, animation_list
        global cont, t, speed
        totalcolor = np.array([[None] * 9] * 6)
        cont = 0
        # animation_list = [None]
        speed = 100
        animation_list = rechange(recover)

        self.canvasclock.delete(tk.ALL)
        self.canvaslog.delete(tk.ALL)
        self.stop_thread(self.myThread)

    '''显示识别的每一面，在getrgb中调用了'''

    def showresult(self, canvas, posx, posy, color):
        canvas.create_rectangle(posx, posy, posx + 30, posy + 30, fill=color[0], outline='black', width=2)
        canvas.create_rectangle(posx + 2 + 30, posy, posx + 2 + 30 * 2, posy + 30, fill=color[1], outline='black',
                                width=2)
        canvas.create_rectangle(posx + 2 + 2 + 30 * 2, posy, posx + 2 + 2 + 30 * 3, posy + 30, fill=color[2],
                                outline='black', width=2)

        canvas.create_rectangle(posx, posy + 2 + 30, posx + 30, posy + 2 + 30 * 2, fill=color[3], outline='black',
                                width=2)
        canvas.create_rectangle(posx + 2 + 30, posy + 2 + 30, posx + 2 + 30 * 2, posy + 2 + 30 * 2, fill=color[4],
                                outline='black', width=2)
        canvas.create_rectangle(posx + 2 + 2 + 30 * 2, posy + 2 + 30, posx + 2 + 2 + 30 * 3, posy + 2 + 30 * 2,
                                fill=color[5], outline='black', width=2)

        canvas.create_rectangle(posx, posy + 2 + 2 + 30 * 2, posx + 30, posy + 2 + 2 + 30 * 3, fill=color[6],
                                outline='black', width=2)
        canvas.create_rectangle(posx + 2 + 30, posy + 2 + 2 + 30 * 2, posx + 2 + 30 * 2, posy + 2 + 2 + 30 * 3,
                                fill=color[7], outline='black', width=2)
        canvas.create_rectangle(posx + 2 + 2 + 30 * 2, posy + 2 + 2 + 30 * 2, posx + 2 + 2 + 30 * 3,
                                posy + 2 + 2 + 30 * 3,
                                fill=color[8], outline='black', width=2)

    '''显示一个色块，与显示一面色块不同'''

    def colorshow(self, color):
        self.canvasresult.create_rectangle(0, 0, 100, 100, fill=color[0], outline='black', width=5)
        self.canvasresult.create_rectangle(103, 0, 203, 100, fill=color[1], outline='black', width=5)
        self.canvasresult.create_rectangle(206, 0, 300, 100, fill=color[2], outline='black', width=5)

        self.canvasresult.create_rectangle(0, 103, 100, 200, fill=color[3], outline='black', width=5)
        self.canvasresult.create_rectangle(103, 103, 203, 200, fill=color[4], outline='black', width=5)
        self.canvasresult.create_rectangle(206, 103, 300, 200, fill=color[5], outline='black', width=5)

        self.canvasresult.create_rectangle(0, 203, 100, 300, fill=color[6], outline='black', width=5)
        self.canvasresult.create_rectangle(103, 203, 203, 300, fill=color[7], outline='black', width=5)
        self.canvasresult.create_rectangle(206, 203, 300, 300, fill=color[8], outline='black', width=5)

    '''K近邻算法'''

    def classify0(self, inX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]  # numpy函数shape[0]返回dataSet的行数
        diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
        # print(diffMat)
        sqDiffMat = diffMat ** 2  # 二维特征相减后平方
        # print(sqDiffMat)
        sqDistances = sqDiffMat.sum(axis=1)  # sum()所有元素相加，sum(0)列相加，sum(1)行相加
        distances = sqDistances ** 0.5  # 开方，计算出距离
        sortedDistIndices = distances.argsort()  # 返回distances中元素从小到大排序后的索引值
        # print(sortedDistIndices)
        classCount = {}  # 定一个记录类别次数的字典
        for i in range(k):  # 取出前k个元素的类别   k是被这个半径包住的点的个数
            voteIlabel = labels[sortedDistIndices[i]]  # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计算类别次数
        # python3中用items()替换python2中的iteritems()
        # key=operator.itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(0)根据字典的键进行排序
        # reverse降序排序字典
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]  # 返回次数最多的类别,即所要分类的类别

    '''从opencv中读得视频然后显示在tkinter上'''

    def showvideo(self):
        global face, frame
        clock_pic_change = [None] * 9
        self.cap = cv2.VideoCapture(0)
        while True:  # 读取视频
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1, dst=None)  # 大小为480*640，宽，高
            frame = frame[100:400, 100:400]
            # frame = cv2.flip(frame, 1, dst=None)  # 大小为480*640，宽，高
            # frame = cv2.line(frame, (0, 0), (511, 511), (255, 0, 0), 5)
            tkImage = self.pic_change(frame)
            self.canvasvedio.create_image(0, 0, anchor='nw', image=tkImage)
            abc = None
            abc = tkImage  # 延迟图片销毁，不然会闪烁g
        # # 绘制采集框
        # self.init_window_name.update()
        # self.init_window_name.after(1)

    '''聚类算法，将每个图片块的颜色进行聚类，得出主要颜色'''

    def color_quantization1(self, image, k):
        # 将图像转换为数据
        data = np.float32(image).reshape((-1, 3))
        # 算法终止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        # K-Means 聚类
        ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # 簇中心
        center = np.uint8(center)
        # 将具有 k 颜色中心的图像转换为 uint8
        result = center[label.flatten()]
        result = result.reshape(image.shape)
        counter = collections.Counter(label.flatten())
        return result, counter, center

    '''调用魔方复原算法，得出魔方解决的步骤'''

    def solver(self):
        global totalcolor
        global st
        global ok, speed, recover
        global animation_list_ani, frontcolor, animation_list_sol, animation_list
        cubelist = np.array([[None] * 9] * 6)
        relist = [None] * 54

        # # 前，右，后，左，上，下
        # # 目标顺序：上，右，前，下，左，后
        # 由于识别的顺序和标准的魔方顺序不一致，需要进行存储位置的调换

        totalcolor[[0, 4], :] = totalcolor[[4, 0], :]  # 上，右，后，左，前，下
        totalcolor[[2, 4], :] = totalcolor[[4, 2], :]  # 上，右，前，左，后，下
        totalcolor[[3, 5], :] = totalcolor[[5, 3], :]  # 上，右，前，下，后，左
        totalcolor[[4, 5], :] = totalcolor[[5, 4], :]  # 上，右，前，下，后，左
        # 得出中心色块
        up = totalcolor[0][4]  # 寻找中心色块，不论中心色块是什么颜色，方位是不会变的
        right = totalcolor[1][4]
        front = totalcolor[2][4]
        down = totalcolor[3][4]
        left = totalcolor[4][4]
        back = totalcolor[5][4]

        clo = front + right + up
        # 前，右，上 动画魔方的方位，这里要识别出这几个面的颜色，以便动画魔方对应起现实魔方
        for i in range(3):
            if clo[i] == '白':
                frontcolor[i] = 'w'
            elif clo[i] == '红':
                frontcolor[i] = 'r'
            elif clo[i] == '绿':
                frontcolor[i] = 'g'
            elif clo[i] == '蓝':
                frontcolor[i] = 'b'
            elif clo[i] == '黄':
                frontcolor[i] = 'y'
            elif clo[i] == '橙':
                frontcolor[i] = 'o'
        # print(frontcolor)

        for i in range(6):  # 将色块转换为魔方状态字符串，因为输入解魔方函数的内容是由方位面组成的字符串，并不是用的颜色
            for j in range(9):
                if totalcolor[i][j] == up:
                    cubelist[i][j] = 'U'
                elif totalcolor[i][j] == right:
                    cubelist[i][j] = 'R'
                elif totalcolor[i][j] == front:
                    cubelist[i][j] = 'F'
                elif totalcolor[i][j] == down:
                    cubelist[i][j] = 'D'
                elif totalcolor[i][j] == left:
                    cubelist[i][j] = 'L'
                elif totalcolor[i][j] == back:
                    cubelist[i][j] = 'B'

        recubelist = cubelist.reshape(1, -1)  # 转换成一行，也就是一条字符串
        recubelist2 = recubelist.tolist()
        for i in range(54):
            relist[i] = recubelist2[0][i]
        st = ''.join(relist)  # 转换成字符串

        ok = sv.solve(st, 19, 2)  # 得到解法
        erro = ok[0]  # 从返回的函数中得到结果状态

        if erro == 'E':  # 返回E说明识别有误，不能得出正确解
            # print('识别出的魔方颜色面有错，请重新识别！')
            showwarning(title='提示', message='识别有误，不能还原！')
        else:
            # print(ok)
            strd = ok.replace(" ", "")
            strd = strd[:-5]
            ll = showrubikecolor(strd)
            animation_list_sol = reslutcahnge(strd)
            kk = changtotherightfront(frontcolor[0], frontcolor[1], frontcolor[2])
            animation_list_ani = kk + ll
            animation_list = animation_list_ani
            recover = kk + ll + animation_list_sol
            speed = 100
            print(speed)
            num = len(strd) // 6  # 算出有几排
            for i in range(num):
                self.canvaslog.create_text(100, i * 40 + 50, text=strd[i * 6:i * 6 + 6], fill='#2F4F4F',
                                           font=('微软雅黑', 15, 'bold'))
            self.canvasclock.create_text(58, 90, text='此面正对你', fill='#2F4F4F', font=('微软雅黑', 10, 'bold'))

    '''展示复原动画'''

    def fuyuan(self):
        global animation_list, speed
        animation_list = animation_list_sol
        speed = 1
        print(speed)

    '''关闭窗口'''

    def clos_window(self):
        ans = askyesno(title='退出', message='是否确定退出程序？')
        if ans:
            self.init_window_name.destroy()
            sys.exit()
        else:
            return None


if __name__ == '__main__':
    """ 把button方法打包进线程，现实运行不卡顿 """
    """ 实例化出一个父窗口 """
    init_window = tk.Tk()
    # """ tk界面置顶 """
    # init_window.attributes("-topmost", 1)
    """ 创建Gui类对象 """
    test_gui = TestGui(init_window)
    """ 初始化GUi组件 """
    init_window.mainloop()
