#coding:utf-8
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
import time
def mail(my_user,repwd):
  ret=True
  my_sender='18636155578@163.com'           #发件人邮箱，需开启smtp服务
  pwd='chaojun123'                         #如是网易，此为客户端授权码
  try:
    str_text='朝俊哥哥躁扰你： '+repwd     #邮件内容(可修改)
    msg=MIMEText(str_text,'plain','utf-8')
    msg['From']=my_sender
    msg['To']=my_user
    msg['Subject']="代码计算完成"         #邮件主题(可修改)
    server=smtplib.SMTP("smtp.163.com",25)
    server.login(my_sender,pwd)
    server.sendmail(my_sender,my_user,msg.as_string())
    server.quit()
  except Exception:
    ret=False
  return ret
def main1():
    my_user = '610123802@qq.com'  # 收件人邮箱(可修改)
    repwd = 'chaochao123...'  # 重置的密码(可修改)
    for i in range(1):
        ret = mail(my_user, repwd)
        if ret:
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "Success")
        else:
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), "Filed")


if __name__ == '__main__':
    main1()
