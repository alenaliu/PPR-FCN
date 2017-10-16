
import cv2, os
import _init_paths
import numpy as np
import utils.zl_utils as zl

def test_edge_box_py():
    import edgebox.edge_boxes as eb
    im = cv2.imread('/home/zawlin/data/indoor.jpg')
    im = cv2.resize(im,(500,640))
    zl.tic()
    windows = eb.get_windows(['/home/zawlin/data/indoor.jpg'],maxboxes=10)
    windows = np.array(windows[0])
    for i in xrange(windows.shape[0]):
        b = windows[i].astype(np.int32)
        cv2.rectangle(im,(b[1],b[0]),(b[3],b[2]),(255,0,0),2)
    cv2.imshow('drawn',im)
    cv2.waitKey(0)

def test_edge_boxes_cpp_wrapper():
    from edge_boxes_python import edge_boxes_python
    eb = edge_boxes_python('edges/cpp/external/gop_1.3/data/sf.dat')
    im = cv2.imread('/home/zawlin/data/indoor.jpg')
    #im = cv2.resize(im,(500,640))
    #im = zl.url_to_image('http://images1.mtv.com/uri/mgid:file:docroot:cmt.com:/sitewide/assets/img/artists/wesley_james/photo_gallery/didnt_i/8-Indoor-Scene-x600.jpg')
    print 'detecting'
    zl.tic()
    bbs = eb.get_edge_boxes(im)
    bbs = bbs[bbs[:,4].argsort()[::-1]]
    for i in xrange(10):
        b = bbs[i]
        cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),(255,0,0),2)
    cv2.imshow('drawn',im)
    cv2.waitKey(0)
    print zl.toc()
    print bbs

def test_matlab_engine():
    import utils.zl_edge_box as zl_eb
    zl.tic()
    windows = zl_eb.get_windows('/home/zawlin/data/indoor.jpg')
    print zl.toc()

    windows = np.array(windows)
    im = cv2.imread('/home/zawlin/data/indoor.jpg')

    scores = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]
    ims = []
    for s in scores:
        ims.append(im.copy())
    for i in xrange(len(scores)):
        s = scores[i]
        im_render = ims[i]
        for i in xrange(windows.shape[0]):
            roi = windows[i].astype(np.int32)
            if windows[i][4] > s:
                cv2.rectangle(im_render,(roi[0],roi[1]),(roi[2],roi[3]),(255,0,0),1)
        cv2.imwrite('/home/zawlin/data/iccv/iccv_thresh_%f.jpg'%s,im_render)
    for i in xrange(windows.shape[0]):
        b = windows[i].astype(np.int32)
        cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),(255,0,0),2)
    cv2.imshow('drawn',im)
    cv2.waitKey(0)

cv2.namedWindow('drawn',0)
#test_edge_boxes_cpp_wrapper()
#test_edge_box_py()
test_matlab_engine()
