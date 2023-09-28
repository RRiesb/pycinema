from PySide6 import QtCore, QtWidgets, QtGui

from pycinema.theater.View import View

from pycinema.theater import views
from pycinema.theater.views.FilterView import FilterView
from pycinema.theater.views.NodeView import NodeView

class SelectionButton(QtWidgets.QPushButton):
  def __init__(self,name,parent,cls):
    super().__init__(name,parent)
    self.cls = cls
    self.clicked.connect(self.replaceView)

  def replaceView(self):
    self.parent().parent().replaceView(self.parent(),self.cls)

class SelectionView(View):
  def __init__(self):
    super().__init__()
    self.setTitle(self.__class__.__name__)

    self.content.layout().addWidget(QtWidgets.QLabel(),1)

    view_list = [cls for name, cls in views.__dict__.items() if isinstance(cls,type) and issubclass(cls,FilterView) and name!='FilterView']
    view_list.sort(key=lambda x: x.__name__)
    view_list.insert(0,NodeView)

    for cls in view_list:
      self.layout().addWidget( SelectionButton(cls.__name__, self, cls) )

    self.layout().addWidget(QtWidgets.QLabel(),1)
