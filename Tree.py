class Tree:

  def __init__(self, value, targetCount):
    self.value = value
    # kalau inner node, value itu attribute, kalau  leaf node, value itu prediksi
    self.targetCount = targetCount
    self.entropy = 0
    self.children = []
    # format target count : (label, jumlah), contoh : ('+',10)

  def getValue(self):
    return self.value

  def getTargetCount(self):
    return self.targetCount

  def getChildren(self):
    return self.children

  def getEntropy(self):
    return self.entropy

  def setValue(self, value):
    self.value = value

  def setTargetCount(self, value):
    self.value = value

  def setEntropy(self, entropy):
    self.entropy = entropy

  def addChildren(self, attributeValue, childTree):
    self.children.append((attributeValue, childTree))

  def checkChildrenValueExist(self, attributeValue):
    found = False
    for c in self.children:
      if c[0] == attributeValue:
        found = True
        break
    
    return found
  def checkChildrenValueSatisfy(self,attributeValue):
    found = False
    if(attributeValue[0]=='<' or attributeValue[0]=='>' or attributeValue[0]=='='):
      for c in self.children:
          string = str(c[0]) + str(attributeValue)
          if eval(string):
              found = True
              break 
    return found

  def gotoMaxChildrenCount(self):
    idx = -1
    maxcount = 0
    for i in range(len(self.targetCount)):
      if self.targetCount[i][1] > maxcount:
        maxcount = self.targetCount[i][1]
        idx = i
    
    return self.children[idx][1]

  def gotoSpesificChildren(self, attributeValue):
    for c in self.children:
      if c[0] == attributeValue:
        return c[1]
    
    return None
      


  def prune(self):
    max = 0
    sum = 0
    for label in targetCount:
      _x, count = label
      sum += count
      if(max <= count):
        max = count
        maxLabel = label
    maxLabel, y = maxLabel
    self.value = maxLabel
    self.targetCount.clear()
    self.targetCount.append((maxLabel, sum))
    self.children.clear()

  def printTree(self, level=1):

    print(self.value, self.targetCount, end="")
    if(self.entropy):
      print(' Entropy : ', self.entropy)
    else:
      print()
    for t in self.children:
      print('\t'*level, t[0], ' -> ', end="")
      t[1].printTree(level+1)


testTree = Tree('Outlook', (('+', 10), ('-', 9)))
testTree.setEntropy(10)
humidity = Tree('Humidity', (('+', 4), ('-', 2)))
noHumidity = Tree('No', (('+', 0), ('-', 2)))
yesHumidity = Tree('Yes', (('+', 4), ('-', 0)))
humidity.addChildren('High', noHumidity)
humidity.addChildren('Normal', yesHumidity)
testTree.addChildren('Sunny', humidity)
# testTree.printTree()

'''contoh tree :
  Outlook(('+',10), ('-',9))
    |Sunny -> Humidity(('+',4), ('-',2))
          |High -> No('+',0), ('-',2)
          |Normal -> Yes(('+',4), ('-',0))
    |Overcast -> Yes(('+',2), ('-',0))
    |Rain -> Wind
          |Strong -> No(('+',0), ('-',5))
          |Weak -> ('+',4), ('-',0)
  (1)
  value : Outlook
  targetCount : (('+',10), ('-',9))
  children = [(Sunny,Tree(Humidity)),(Overcast,Tree(Yes+2,0-)),(Rain,Tree(Wind))]
  entropy = itung pake rumus arvin wkwkwk
'''