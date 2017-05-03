"use strict"
const express = require('express')
const app = express()
const http = require('http').Server(app)
const io = require('socket.io')(http)
const spawn = require('child_process').spawn
const bodyParser = require('body-parser')
const mkdirp = require('mkdirp');
const fs = require('fs');
const glob = require('glob');

const clients = {}
const users = {
  'ms2666': { userId: 51, name: 'Mukund', pic: '' },
  'fc249': { userId: 52, name: 'Frank', pic: 'https://media.licdn.com/mpr/mpr/shrinknp_200_200/AAEAAQAAAAAAAASHAAAAJGE4MTI5Mzk0LWJmN2QtNDE3NS1hNzIzLTFkZmM2YmQxMTM1Mg.jpg' }
}

app.use(bodyParser.text({
  type: 'text/plain',
  limit: '4mb'
}))

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html')
})

app.post('/login/:id', (req, res) => {
  console.log('LOGIN ROUTE REACHED')
  // id is the client we want to connect to
  if (!(req.params.id in clients)) {
    return res.status(404).json({ message: 'Client with ID: ' + req.params.id + ' not found.'})
  }

  let id = req.params.id

  // plaintext request to log in based on this
  let dir = './Data_test'
  let path = 'u000_w000'
  let filename = 'u000_w000_accelerometer.log'

  console.log('Checking if RUNNING file exists')

  while(fs.existsSync('./Data_test/RUNNING')) {}

  console.log("No RUNNING file detected")
  
  if (fs.existsSync('./Data_test/RESULT')) {
    fs.unlinkSync('./Data_test/RESULT')
  }

  writeToFile(dir, path, filename, req.body, function (err) {
    if (err) {
      return res.status(500).json({ message: 'Disaster! Internal Server Error' })
    }
    // Wait on the RESULT file
    waitOnResult(function(result) {
      console.log("RESULT FILE FOUND")
      authenticate(result, io, id)
      fs.unlinkSync('./Data_test/RESULT')
      return res.status(200).json({ attempted: true })
    })
  })
})

app.post('/train/:id', (req, res) => {
  if (!(req.params.id in users)) {
    return res.status(404).json({ message: 'User with ID: ' + req.params.id + ' not found.'})
  }

  console.log(req.params.id)
  console.log(users[req.params.id])
  formNextFileName(true, users[req.params.id].userId, function (err, filename) {
    let dir = './Data'
    let path = filename
    // Let's get how many files there are...
    filename = filename + '_accelerometer.log'
    writeToFile(dir, path, filename, req.body, function (err) {
      if (err) {
        return res.status(500).json({ message: 'Disaster! Internal Server Error' })
      }
      return res.status(200).json({ message: 'Data posted!' })
    })
  })
})

io.on('connection', (socket) => {
  console.log('A client connected. Registering client...')
  registerClient(socket, (id) => {
    console.log('Client registered with id ' + id + '. Returning new client id.')
    // emit this id back to the client
    socket.emit('client_id', id)
  })

  // define the socket handlers
  socket.on('disconnect', () => {
    // how do we get an id
    // unregisterClient(socket, id)
  })

  socket.on('login', (id) => {
    console.log('Received login')
    setTimeout(function() {
      console.log('Doing work')
      attemptLogin(io, id)
    }, 0)
  })
})

app.use(express.static(__dirname))

http.listen(3000, () => {
  console.log('listening on *:3000')
})

function registerClient(socket, callback) {
  if (!callback || typeof callback !== "function") {
    throw new Error("registerClient(socket, callback) expected callback to be a function")
  }

  let id = 0
  do {
    id = Math.floor(Math.random() * 1000 + 1)
  } while (id in clients);

  // register this socket to a room with this name
  socket.join(id, () => {
    clients[id] = 'val'
    callback(id);
  })
}

function unregisterClient(socket, id) {
  // unregister this socket
  socket.leave(id, () => {
    delete clients[id]
  })
}

function authenticate(person, io, id) {
  console.log("Trying to authenticate person: " + person)
  if (person !== 'other') {
    io.to(id).emit('access', users[person] )
  } else {
    io.to(id).emit('no-access')
  }
}

function formNextFileName(training, userId, callback) {
  if (!callback || typeof callback !== "function") {
    throw new Error("writeToFile(path, filename, contents, callback) expected callback to be a function")
  }
  // leftpad with 0's
  userId = zfill3(userId)
  if (training) {
    console.log('Search string: ' + './Data/u' + userId + '*')
    glob('./Data/u' + userId + '*', function(err, files) {
      if (err) {
        callback(err)
      }
      let walkId = files.length + 1
      walkId = zfill3(walkId)
      callback(null, 'u' + userId + '_w' + walkId)
    })
  } else {
    walkId = '000'
    callback(null, 'u' + userId + '_w' + walkId)
  }
}

function zfill3(num) {
  return ("000" + num).slice(-3)
}

function writeToFile(dir, path, filename, contents, callback) {
  if (!callback || typeof callback !== "function") {
    throw new Error("writeToFile(path, filename, contents, callback) expected callback to be a function")
  }
  mkdirp(dir + '/' + path, function (err) {
    fs.open(dir + '/' + path + '/' + filename,'r',function (err, fd) {
      if (err) {
        fs.writeFile(dir + '/' + path + '/' + filename, contents, function (err) {
            if(err) {
              callback(err)
            }
            console.log("File " + filename + " created!")
            callback(null)
        })
      } else {
        console.log("The file already exists!")
        callback(new Error("file already exists"))
      }
    })
  })
}

function waitOnResult(callback) {
  while(!fs.existsSync('./Data_test/RESULT')) { }
  fs.readFile('./Data_test/RESULT', function (err, data) {
    if (err) {
      console.log('Error: ' + err)
    }
    data = String.fromCharCode.apply(null, new Uint16Array(data)).trim()
    console.log('Result is: ' + data)
    callback(data)
  })
}
